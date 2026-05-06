#!/usr/bin/env python3
import sys
import yaml
import os
import torch
import gpytorch
import numpy as np
import pandas as pd

# =========================================================
# 讀取參數（改成 -c config_dir）
# =========================================================
if len(sys.argv) < 3 or sys.argv[1] != "-c":
    print("[Error] Usage: python check_MSPE.py -c <config_file>")
    sys.exit(1)

config_dir = sys.argv[2]  # model.yaml 所在的位置

if not os.path.isdir(config_dir):
    print(f"[Error] Config directory not found: {config_dir}")
    sys.exit(1)

# =========================================================
# 讀取 model.yaml → 自動推論 filename
# =========================================================
model_yaml = os.path.join(config_dir, "model.yaml")
if not os.path.isfile(model_yaml):
    print(f"[Error] Cannot find: {model_yaml}")
    sys.exit(1)

with open(model_yaml, "r") as f:
    model_cfg = yaml.safe_load(f)

filename            = model_cfg["output_name"]
output_dir          = model_cfg["output_directory"]
dataset_dir         = model_cfg["dataset_directory"]

# 推出 ROOT
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
result_base = os.path.join(ROOT, output_dir, filename)
os.makedirs(result_base, exist_ok=True)

data_root = os.path.join(ROOT, dataset_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(os.path.join(data_root, "data_train_known_real.npy"), allow_pickle=True)[:, :, 0]   # → (160, 448)
loc_known = np.load(os.path.join(data_root, "stations_known_locations.npy"), allow_pickle=True)    # (160, 2)
t_train = np.load(os.path.join(data_root, "hour_list_train.npy"), allow_pickle=True)               # (448,)
loc_unknown = np.load(os.path.join(data_root, "stations_unknown_locations.npy"), allow_pickle=True)
t_test = np.load(os.path.join(data_root, "hour_list_test.npy"), allow_pickle=True)

t_train = np.arange(len(t_train)) + 1
t_test = np.arange(len(t_train), len(t_train) +len(t_test)) + 1

N_known, T_train = data.shape
N_unknown = loc_unknown.shape[0]
T_test = t_test.shape[0]


# 正規化時間：避免 kernel 受時間尺度影響
t_train_norm = (t_train - t_train.min()) / (t_train.max() - t_train.min())
t_test_norm  = (t_test  - t_train.min()) / (t_train.max() - t_train.min())

# spatial normalize
loc_mean = loc_known.mean(axis=0)
loc_std = loc_known.std(axis=0)

loc_known = (loc_known - loc_mean) / loc_std
loc_unknown = (loc_unknown - loc_mean) / loc_std

# ==============================================================
#  建立訓練座標 (160*448, 3)  → [x, y, time]
# ==============================================================

# 空間重複時間次數
loc_rep = np.repeat(loc_known[:, None, :], T_train, axis=1)
# 時間重複空間次數
time_rep = np.repeat(t_train_norm[None, :, None], N_known, axis=0)

X_train = np.concatenate([
    loc_rep.reshape(-1, 2),
    time_rep.reshape(-1, 1)
], axis=1)  # → (160*448, 3)

y_train = data.reshape(-1)  # → (160*448,)

# to torch
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

# ==============================================================
#  定義 Spatiotemporal GP Kernel
# ==============================================================

class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=3)
        )

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)
# ==========================================================
# 🔥 IMPORTANT: inducing points (FIX HERE)
# ==========================================================
num_inducing = 1024  # 可調 256 / 512 / 1024

idx = torch.randperm(X_train.size(0))[:num_inducing]
inducing_points = X_train[idx].clone()   # ✔ 必須 clone + float + device
# ==============================================================
#  訓練 SVGP
# ==============================================================

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = SVGPModel(inducing_points).to(device)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ✅ SVGP 必須用 VariationalELBO
mll = gpytorch.mlls.VariationalELBO(
    likelihood,
    model,
    num_data=y_train.size(0)
)

EPOCHS = 500
for i in range(EPOCHS):
    optimizer.zero_grad()

    output = model(X_train)

    loss = -mll(output, y_train)

    loss.backward()
    optimizer.step()

    if (i + 1) % 5 == 0:
        print(f"[Epoch {i+1}/{EPOCHS}] Loss = {loss.item():.4f}")


# ==============================================================
#  推論函式：任意 (loc, time) 補值
# ==============================================================

def gp_predict(loc_array, t_array_norm):
    """
    loc_array: (N_loc, 2)
    t_array_norm: (T,)
    return shape: (N_loc, T)
    """
    N_loc = loc_array.shape[0]
    T = t_array_norm.shape[0]

    loc_rep = np.repeat(loc_array[:, None, :], T, axis=1)
    time_rep = np.repeat(t_array_norm[None, :, None], N_loc, axis=0)

    X = np.concatenate([loc_rep.reshape(-1, 2), time_rep.reshape(-1, 1)], axis=1)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        pred = likelihood(model(X)).mean

    return pred.reshape(N_loc, T).cpu().numpy()
# ==============================================================
#  1. 補已知地點未來 → (160, T_test)
# ==============================================================
pred_known_future = gp_predict(loc_known, t_test_norm)
#np.save("gp_known_future.npy", pred_known_future)
print("✔ 已知地點未來補完 gp_known_future.npy")


# ==============================================================
#  2. 補未知地點過去 → (N_unknown, 448)
# ==============================================================
pred_unknown_past = gp_predict(loc_unknown, t_train_norm)
#np.save("gp_unknown_past.npy", pred_unknown_past)
print("✔ 未知地點過去補完 gp_unknown_past.npy")


# ==============================================================
#  3. 補未知地點未來 → (N_unknown, T_test)
# ==============================================================
pred_unknown_future = gp_predict(loc_unknown, t_test_norm)
#np.save("gp_unknown_future.npy", pred_unknown_future)
print("✔ 未知地點未來補完 gp_unknown_future.npy")


y_inf = np.concatenate([np.concatenate([data, pred_unknown_past], axis=0), np.concatenate([pred_known_future, pred_unknown_future], axis=0)], axis=1)
y_inf = y_inf.T[np.newaxis, :, :]

# =========================================================
# save results
# =========================================================
pred_filename = f"1 channel.npy"
save_path = os.path.join(result_base, pred_filename)
np.save(save_path, y_inf)
print(f"Saved to: {save_path}")

# =========================================================
# Load real data
# =========================================================
test_data = np.load(os.path.join(data_root, "data_real.npy"), allow_pickle=True)

hour_list_train = np.load(os.path.join(data_root, "hour_list_train.npy"), allow_pickle=True)
hour_list_test = np.load(os.path.join(data_root, "hour_list_test.npy"), allow_pickle=True)
hour_list = np.concatenate([hour_list_train, hour_list_test], axis=0)
future_days = hour_list_test.shape[0]

past_idx = np.arange(test_data.shape[1])[:-future_days]
future_idx = np.arange(test_data.shape[1])[-future_days:]

unknown_amount = np.load(os.path.join(data_root, "stations_unknown_locations.npy"), allow_pickle=True).shape[0]
unknown_idx = np.arange(test_data.shape[2])[-unknown_amount:]
known_idx = np.arange(test_data.shape[2] - unknown_amount)

y_inf = y_inf.astype(np.float64)
real = test_data.astype(np.float64)

# =========================================================
# metrics
# =========================================================
def mspe(pred, true):
    return np.mean((pred - true) ** 2)

# =========================================================
# 計算三個目標指標
# =========================================================
Unobs_Future = mspe(y_inf[:, future_idx, :][:, :, unknown_idx],
                    real[:, future_idx, :][:, :, unknown_idx])

Unobs_Past = mspe(y_inf[:, past_idx, :][:, :, unknown_idx],
                  real[:, past_idx, :][:, :, unknown_idx])

Obs_Future = mspe(y_inf[:, future_idx, :][:, :, known_idx],
                  real[:, future_idx, :][:, :, known_idx])

# =========================================================
# 寫入 CSV（新增一行）
# =========================================================
csv_path = os.path.join(ROOT, "autorun", f"{filename}.csv")

if not os.path.isfile(csv_path):
    print(f"[Error] Missing CSV: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

No = len(df) + 1  # append 前行數

new_row = {
    "No.": No,
    "Unobs Future": Unobs_Future,
    "Unobs Past": Unobs_Past,
    "Obs Future": Obs_Future,
}

# ===== 自動兼容 pandas 1.x / 2.x =====
try:
    # pandas 1.x
    df = df.append(new_row, ignore_index=True)
except AttributeError:
    # pandas 2.x
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
df.to_csv(csv_path, index=False)

print("[check_MSPE] Append:", new_row)
