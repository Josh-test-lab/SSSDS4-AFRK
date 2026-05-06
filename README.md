# SSSDS4-AFRK

This is the modified version of [https://github.com/egpivo/SSSD_CP](https://github.com/egpivo/SSSD_CP).

安裝請使用 [envs/conda/build_conda_env.sh](envs/conda/build_conda_env.sh) 。

目錄與檔案排列方式均此相同
https://github.com/egpivo/SSSD_CP

安裝方式可以參考如下
```bash
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /home/u6025091
export PATH="/home/u6025091/bin:$PATH"

conda --version

sudo apt-get update
sudo apt-get install dos2unix

find ~/SSSD_CP/ -type f -exec dos2unix {} \;

pip install poetry
poetry install

cd SSSD_CP/
chmod +x envs/vm/install_on_ubuntu.sh
chmod +x envs/conda/build_conda_env.sh
./envs/vm/install_on_ubuntu.sh
./envs/conda/build_conda_env.sh
```

修改的檔案均在script與sssd中，請參考sssd\training\utils.py

## Usage
```bash
./scripts/autorun/run_job.sh -c configs/configs-MERRA2-1var-S4+AFRK -t 30
```
`-c` for config paths, `-t` for experiment times