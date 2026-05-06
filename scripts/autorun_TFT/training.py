#!/usr/bin/env python3
import os
import sys
import yaml
import numpy as np
from darts.models.forecasting.tft_model import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from tqdm import tqdm
from autoFRK import AutoFRK
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

input_chunk_length  = model_cfg["input_chunk_length"]
hidden_size         = model_cfg["hidden_size"]
lstm_layers         = model_cfg["lstm_layers"]
dropout             = model_cfg["dropout"]
batch_size          = model_cfg["batch_size"]
n_epochs            = model_cfg["n_epochs"]
lr                  = model_cfg["lr"]
add_relative_index  = model_cfg["add_relative_index"]
random_state        = None if model_cfg["random_state"] == "None" else model_cfg["random_state"]
ic                  = model_cfg["ic"]
hours_a_day         = model_cfg["hours_a_day"]

filename            = model_cfg["output_name"]
output_dir          = model_cfg["output_directory"]
dataset_dir         = model_cfg["dataset_directory"]

# 推出 ROOT
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
result_base = os.path.join(ROOT, output_dir, filename)
os.makedirs(result_base, exist_ok=True)

dataset_path = os.path.join(ROOT, dataset_dir)


# =========================================================
# Training function
# =========================================================
def tft_1_channel(
    train_ts,
    n_forecast: int,
    input_chunk_length: int = 80,
    hidden_size: int = 32,
    lstm_layers: int = 1,
    dropout: float = 0.1,
    batch_size: int = 32,
    n_epochs: int = 4000,
    lr: float = 1e-3,
    past_cov = None,
    future_cov = None,
    add_relative_index = False,
    random_state: int = 42,
):
    """
    使用 TFT 訓練並預測未來 n_forecast 期。

    Parameters
    ----------
    train_ts : TimeSeries
        訓練資料（必須是 darts TimeSeries）
    n_forecast : int
        要預測的未來步數
    input_chunk_length : int
        過去序列窗口長度
    output_chunk_length : int
        模型每次輸出的窗口長度
    hidden_size : int
        TFT 隱層維度
    lstm_layers : int
        LSTM 堆疊層數
    dropout : float
        dropout rate
    batch_size : int
        批大小
    n_epochs : int
        訓練 epochs
    lr : float
        學習率
    add_cov : dict
        額外特徵，例如：
        {
            "past": TimeSeries,
            "future": TimeSeries
        }
    random_state : int
        隨機種子

    Returns
    -------
    pred : TimeSeries
        未來 n_forecast 的預測序列
    scaler : Scaler
        資料標準化器
    model : TFTModel
        已訓練的模型
    """

    assert train_ts.ndim == 3, "train_data 必須是 3D array"
    assert train_ts.shape[0] == 1, "train_data 第一維必須是 1"
    
    if np.isnan(train_ts).any():
        raise ValueError("train_data 有 NaN，請先補值或移除。")
    
    add_relative_index = True if future_cov is None else add_relative_index

    # ------------------------------
    # 1. Normalize
    # ------------------------------
    # 1️⃣ target scaler
    scaler_target = Scaler()
    train_scaled = scaler_target.fit_transform(TimeSeries.from_values(train_ts[0]))

    # 2️⃣ past_cov scaler（如果有）
    scaler_past = None
    past_cov_scaled = None
    if past_cov is not None:
        scaler_past = Scaler()
        past_cov_scaled = scaler_past.fit_transform(TimeSeries.from_values(past_cov[0]))

    # 3️⃣ future_cov scaler（如果有）
    scaler_future = None
    future_cov_scaled = None
    if future_cov is not None:
        scaler_future = Scaler()
        future_cov_scaled = scaler_future.fit_transform(TimeSeries.from_values(future_cov[0]))

    # ------------------------------
    # 2. Build TFT Model
    # ------------------------------
    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=n_forecast,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
        optimizer_kwargs={"lr": lr},
        random_state=random_state,
        add_relative_index=add_relative_index,
    )

    # ------------------------------
    # 3. Fit Model
    # ------------------------------
    model.fit(
        series=train_scaled,
        past_covariates=past_cov_scaled,
        future_covariates=future_cov_scaled,
        verbose=True,
    )

    # ------------------------------
    # 4. Predict
    # ------------------------------
    pred_scaled = model.predict(
        n=n_forecast,
        past_covariates=past_cov_scaled,
        future_covariates=future_cov_scaled,
    )

    pred = scaler_target.inverse_transform(pred_scaled).values()

    return pred[np.newaxis, :, :], (scaler_target, scaler_past, scaler_future), model

# =========================================================
# Load real data
# =========================================================
# date 
hour_list_train = np.load(os.path.join(dataset_path, 'hour_list_train.npy'), allow_pickle=True)
hour_list_test = np.load(os.path.join(dataset_path, 'hour_list_test.npy'), allow_pickle=True)
hour_list = np.concatenate([hour_list_train, hour_list_test], axis=0)
future_days = hour_list_test.shape[0]
n_forecast = int(future_days / hours_a_day)

# load data
# train_data: (n_var, n_train_time, n_locs)
# test_data : (n_var, n_test_time, n_locs)
train_data = np.load(os.path.join(dataset_path, 'data_train_known_real.npy'), allow_pickle=True).transpose(2, 1, 0)
real_full_data = np.load(os.path.join(dataset_path, 'data_real.npy'), allow_pickle=True)
future_data = np.load(os.path.join(dataset_path, 'data_test_known_real.npy'), allow_pickle=True).transpose(2, 1, 0)

# locations
stations_known_locations = np.load(os.path.join(dataset_path, 'stations_known_locations.npy'), allow_pickle=True)
stations_unknown_locations = np.load(os.path.join(dataset_path, 'stations_unknown_locations.npy'), allow_pickle=True)

# compare
past_idx = np.arange(real_full_data.shape[1])[:-future_days]
future_idx = np.arange(real_full_data.shape[1])[-future_days:]
unknown_amount = np.load(os.path.join(dataset_path, 'stations_unknown_locations.npy'), allow_pickle=True).shape[0]
unknown_idx = np.arange(real_full_data.shape[2])[-unknown_amount:]
known_idx = np.arange(real_full_data.shape[2])[:real_full_data.shape[2] - unknown_amount]

# =========================================================
# Training
# =========================================================
TFT_pred, scaler, model = tft_1_channel(
    train_ts            = train_data,
    n_forecast          = n_forecast,
    input_chunk_length  = input_chunk_length,
    hidden_size         = hidden_size,
    lstm_layers         = lstm_layers,
    dropout             = dropout,
    batch_size          = batch_size,
    n_epochs            = n_epochs,
    lr                  = lr,
    add_relative_index  = add_relative_index,
    random_state        = random_state
)

TFT_pred = np.concatenate([train_data, TFT_pred], axis = 1)
sp_mean = np.mean(TFT_pred, axis=2, keepdims=True)
sp_std = np.std(TFT_pred, axis=2, keepdims=True)
TFT_pred = (TFT_pred - sp_mean) / (sp_std + 1e-8)

# AFRK
AFRK_pred = np.zeros((TFT_pred.shape[0], TFT_pred.shape[1], unknown_amount))
frk = AutoFRK()
mrts = None
choosen_k = []
for variable in tqdm(range(train_data.shape[0]), desc=f"inferencing autoFRK"):
    data_slice = TFT_pred[variable, :, :].T
    _ = frk.forward(
        data=data_slice,
        loc=stations_known_locations,
        G = mrts,
        requires_grad=False
    )
    fpred = frk.predict(
        newloc = stations_unknown_locations
    )['pred.value']
    mrts = frk.obj['G'] if mrts is None else mrts
    choosen_k.append(frk.obj['G']['MRTS'].shape[1])
    
    AFRK_pred[variable, :, :] = fpred.T.cpu().numpy()
pred = np.concatenate([TFT_pred, AFRK_pred], axis=2)

pred = pred * sp_std + sp_mean

# =========================================================
# save results
# =========================================================
pred_filename = f"1 channel.npy"
save_path = os.path.join(result_base, pred_filename)
np.save(save_path, pred)
print(f"Saved to: {save_path}")


# =========================================================
# metrics
# =========================================================
def mspe(pred, true):
    return np.mean((pred - true) ** 2)

# =========================================================
# 計算三個目標指標
# =========================================================
y_inf = pred
real = real_full_data
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