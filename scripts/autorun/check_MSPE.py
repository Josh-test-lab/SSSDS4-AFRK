#!/usr/bin/env python3
import os
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================================================
# 讀取參數（改成 -c config_dir）
# =========================================================
if len(sys.argv) < 3 or sys.argv[1] != "-c":
    print("[Error] Usage: python check_MSPE.py -c <config_dir>")
    sys.exit(1)

config_dir = sys.argv[2]  # model.yaml / training.yaml 所在的位置

if not os.path.isdir(config_dir):
    print(f"[Error] Config directory not found: {config_dir}")
    sys.exit(1)

# =========================================================
# 讀取 training.yaml → 自動推論 filename
# =========================================================
training_yaml = os.path.join(config_dir, "training.yaml")
if not os.path.isfile(training_yaml):
    print(f"[Error] Cannot find: {training_yaml}")
    sys.exit(1)

with open(training_yaml, "r") as f:
    train_cfg = yaml.safe_load(f)

# output_directory → filename（最後一層）
output_dir = train_cfg["output_directory"].strip("\"'")
filename = os.path.basename(output_dir)

# 推出 ROOT
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
result_base = os.path.join(ROOT, "results", filename)

# =========================================================
# 讀取 diffusion (model.yaml)
# =========================================================
model_yaml = os.path.join(config_dir, "model.yaml")
if not os.path.isfile(model_yaml):
    print(f"[Error] Cannot find: {model_yaml}")
    sys.exit(1)

with open(model_yaml, "r") as f:
    model_cfg = yaml.safe_load(f)

T = model_cfg["diffusion"]["T"]
beta_0 = model_cfg["diffusion"]["beta_0"]
beta_T = model_cfg["diffusion"]["beta_T"]

sub_dir = f"T{T}_beta0{beta_0}_betaT{beta_T}"

# =========================================================
# real_path 來源：train_path 的資料夾
# =========================================================
train_path = train_cfg["data"]["train_path"].strip("\"'")
real_path = os.path.dirname(train_path)

# =========================================================
# result_path
# =========================================================
result_path = os.path.join(result_base, "inference", sub_dir, "max")

# =========================================================
# function: inference_data_concatenate
# =========================================================
def inference_data_concatenate(result_path):
    result_data = []
    file_count = len([
        f for f in os.listdir(result_path)
        if os.path.isfile(os.path.join(result_path, f))
    ])
    if "choosen_k.csv" in os.listdir(result_path):
        file_count -= 1

    for i in tqdm(range(file_count), desc="Loading Inference .npy"):
        path = os.path.join(result_path, f"imputation{i}.npy")
        temp = np.load(path)
        result_data.append(temp)

    return np.concatenate(result_data, axis=0)

# =========================================================
# Load inference
# =========================================================
y_inference = inference_data_concatenate(result_path).transpose(2, 1, 0)

# =========================================================
# Load real data
# =========================================================
test_data = np.load(os.path.join(real_path, "data_real.npy"), allow_pickle=True)

hour_list_train = np.load(os.path.join(real_path, "hour_list_train.npy"), allow_pickle=True)
hour_list_test = np.load(os.path.join(real_path, "hour_list_test.npy"), allow_pickle=True)
hour_list = np.concatenate([hour_list_train, hour_list_test], axis=0)
future_days = hour_list_test.shape[0]

past_idx = np.arange(test_data.shape[1])[:-future_days]
future_idx = np.arange(test_data.shape[1])[-future_days:]

unknown_amount = np.load(os.path.join(real_path, "stations_unknown_locations.npy"), allow_pickle=True).shape[0]
unknown_idx = np.arange(test_data.shape[2])[-unknown_amount:]
known_idx = np.arange(test_data.shape[2] - unknown_amount)

y_inf = y_inference.astype(np.float64)
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

No = len(df) + 1   # append 前行數

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