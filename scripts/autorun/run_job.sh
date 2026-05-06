#!/bin/bash
# Author: Hsu, Yao-Chih
# Usage: bash ./scripts/autorun/run_job.sh -c <config_dir> -t <times>
set -euo pipefail

FG_YELLOW='\033[33m'
FG_RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  echo "Usage: $0 -c <config_dir> -t <times>"
  exit 1
}

config_dir=""
total_times=""

while getopts ":c:t:" opt; do
  case "$opt" in
    c) config_dir="$OPTARG" ;;
    t) total_times="$OPTARG" ;;
    *) usage ;;
  esac
done

[[ -z "$config_dir" || -z "$total_times" ]] && usage
[[ "$total_times" =~ ^[0-9]+$ ]] || { echo "[Error] -t must be a positive integer"; exit 1; }

if [[ "$config_dir" = /* ]]; then
  CONFIG_DIR="$config_dir"
else
  CONFIG_DIR="$ROOT_DIR/$config_dir"
fi

TRAINING_YAML="$CONFIG_DIR/training.yaml"
MODEL_YAML="$CONFIG_DIR/model.yaml"
INFERENCE_YAML="$CONFIG_DIR/inference.yaml"

[[ -f "$TRAINING_YAML" ]]   || { echo "[Error] Missing: $TRAINING_YAML"; exit 1; }
[[ -f "$MODEL_YAML" ]]      || { echo "[Error] Missing: $MODEL_YAML"; exit 1; }
[[ -f "$INFERENCE_YAML" ]]  || { echo "[Error] Missing: $INFERENCE_YAML"; exit 1; }

# 1) 讀取 output_directory 的最後一段作為 filename
output_dir="$(
  awk -F': *' '
    /^[[:space:]]*output_directory:/ {
      val = $2
      sub(/#.*/, "", val)   # 清除從 # 開始之後的內容
      gsub(/^[ \t"]+|[ \t"]+$/, "", val)  # 去掉頭尾空白與引號
      print val
      exit
    }
  ' "$TRAINING_YAML"
)"

if [[ -z "$output_dir" ]]; then
  echo "[Error] output_directory not found in $TRAINING_YAML"
  exit 1
fi

# 去掉前後引號與結尾斜線
output_dir="${output_dir%\"}"
output_dir="${output_dir#\"}"
output_dir="${output_dir%\'}"
output_dir="${output_dir#\'}"
output_dir="${output_dir%/}"

filename="$(basename "$output_dir")"
[[ -n "$filename" ]] || { echo "[Error] Cannot parse filename from output_directory"; exit 1; }

# 2) 準備 autorun CSV
AUTORUN_DIR="$ROOT_DIR/autorun"
mkdir -p "$AUTORUN_DIR"

CSV_FILE="$AUTORUN_DIR/${filename}.csv"

if [[ ! -f "$CSV_FILE" || ! -s "$CSV_FILE" ]]; then
  printf 'No.,Unobs Future,Unobs Past,Obs Future\n' > "$CSV_FILE"
  count=1
else
  count="$(wc -l < "$CSV_FILE" | tr -d ' ')"
fi

start_ts="$(date +%s)"

while true; do
  # 檢查是否已超過次數（例如 CSV 行數 > total_times）
  if (( count > total_times )); then
    end_ts="$(date +%s)"
    elapsed=$((end_ts - start_ts))
    printf '%b[Autorun Finish]%b\n' "$FG_YELLOW" "$FG_RESET"
    echo "[Autorun] Total elapsed: ${elapsed}s"
    break
  fi

  echo "[Autorun] Run #$count / $total_times"

  # 3) training
  bash "$ROOT_DIR/scripts/diffusion/training_job.sh" \
    -m "$MODEL_YAML" \
    -t "$TRAINING_YAML"

  # 4) inference
  bash "$ROOT_DIR/scripts/diffusion/inference_job.sh" \
    -m "$MODEL_YAML" \
    -i "$INFERENCE_YAML"

  # 5) check_MSPE
  python "$ROOT_DIR/scripts/autorun/check_MSPE.py" -c $config_dir

  # 6) rename results folder
  src_dir="$ROOT_DIR/results/$filename"
  dst_dir="$ROOT_DIR/results/${filename}-${count}"

  if [[ ! -d "$src_dir" ]]; then
    echo "[Error] Missing result folder: $src_dir"
    exit 1
  fi

  if [[ -e "$dst_dir" ]]; then
    rm -rf "$dst_dir"
  fi
  mv "$src_dir" "$dst_dir"

  count=$((count + 1))
done