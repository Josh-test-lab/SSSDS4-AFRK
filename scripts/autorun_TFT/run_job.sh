#!/bin/bash
# Author: Hsu, Yao-Chih
# Usage: bash ./scripts/autorun_TFT/run_job.sh -c <config_dir> -t <times>
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

MODEL_YAML="$CONFIG_DIR/model.yaml"
[[ -f "$MODEL_YAML" ]]      || { echo "[Error] Missing: $MODEL_YAML"; exit 1; }

# 直接讀取 output_name
filename="$(
  awk -F': *' '
    /^[[:space:]]*output_name:/ {
      val = $2
      sub(/#.*/, "", val)                     # 去掉 # 後面的註解
      gsub(/^[ \t"]+|[ \t"]+$/, "", val)     # 去掉前後空白和引號
      print val
      exit
    }
  ' "$MODEL_YAML"
)"

if [[ -z "$filename" ]]; then
  echo "[Error] output_name not found in $MODEL_YAML"
  exit 1
fi

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
  conda run -n darts python -u "$ROOT_DIR/scripts/autorun_TFT/training.py" -c $config_dir

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