import argparse
from typing import Optional, Union

import os  # New
import numpy as np
import torch
import torch.nn as nn
from autoFRK import MRTS
import yaml

from sssd.core.model_specs import MODEL_PATH_FORMAT, setup_model
from sssd.data.utils import get_dataloader
from sssd.inference.generator import DiffusionGenerator
from sssd.utils.logger import setup_logger
from sssd.utils.utils import calc_diffusion_hyperparams, display_current_time

LOGGER = setup_logger()


def fetch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        default="configs/model.yaml",
        help="Model configuration",
    )
    parser.add_argument(
        "-i",
        "--inference_config",
        type=str,
        default="configs/inference_config.yaml",
        help="Inference configuration",
    )
    parser.add_argument(
        "-ckpt_iter",
        "--ckpt_iter",
        default="max",
        help='Which checkpoint to use; assign a number or "max" to find the latest checkpoint',
    )
    return parser.parse_args()


def run_job(
    model_config: dict,
    inference_config: dict,
    device: Optional[Union[torch.device, str]],
    ckpt_iter: Union[str, int],
) -> None:
    trials = inference_config.get("trials")
    batch_size = inference_config["batch_size"]

    local_path = MODEL_PATH_FORMAT.format(
        T=model_config["diffusion"]["T"],
        beta_0=model_config["diffusion"]["beta_0"],
        beta_T=model_config["diffusion"]["beta_T"],
    )

    diffusion_hyperparams = calc_diffusion_hyperparams(
        **model_config["diffusion"], device=device
    )
    LOGGER.info(display_current_time())

    location_path=os.path.abspath(inference_config["known_location_path"])
    loc = torch.from_numpy(np.load(location_path)).to(dtype=torch.float32)
    use_mrts=model_config['MRTS'].get("use_mrts", False)
    mrts_dim = model_config['MRTS'].get("mrts_dim", -1)
    N = torch.tensor(loc.shape[0], dtype=torch.float32, device=device)
    klim = torch.minimum(N, torch.round(10 * torch.sqrt(N))).to(torch.int64)
    if use_mrts:
        assert mrts_dim == -1 or mrts_dim > 0, "MRTS dimension must be -1 (for default) or a positive integer."
        if mrts_dim == -1:
            pass  # klim is already set to the default value based on N
        else:
            klim = torch.tensor(mrts_dim, dtype=torch.float32, device=device)
        LOGGER.info(f"Using MRTS with {klim} dimensions for spatial training.")

    net = setup_model(inference_config["use_model"], model_config, use_mrts=use_mrts, mrts_dim=klim, device=device)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 0:
        net = nn.DataParallel(net)

    data_names = ["imputation", "original", "mask"]
    directory = inference_config["output_directory"]

    if trials > 1:
        directory += "_{trial}"

    for trial in range(1, trials + 1):
        LOGGER.info(f"The {trial}th inference trial")
        saved_data_names = data_names if trial == 0 else data_names[0]

        DiffusionGenerator(
            net=net,
            device=device,
            diffusion_hyperparams=diffusion_hyperparams,
            local_path=local_path,
            data_path=inference_config["data"]["test_path"],  # New
            output_directory=directory.format(trial=trial) if trials > 1 else directory,
            ckpt_path=inference_config["ckpt_path"],
            ckpt_iter=ckpt_iter,
            batch_size=batch_size,
            masking=inference_config["masking"],
            missing_k=inference_config["missing_k"],
            only_generate_missing=inference_config["only_generate_missing"],
            saved_data_names=saved_data_names,
            use_mrts=use_mrts,  # New
            mrts_dim=klim,  # New
            enable_spatial_prediction=inference_config.get("enable_spatial_inference", True),  # New
            enable_spatial_normalization=inference_config.get("enable_spatial_normalization", True),  # New
            known_location_path=os.path.abspath(inference_config["known_location_path"]),  # New
            unknown_location_path=os.path.abspath(inference_config["unknown_location_path"]),  # New
            AFRK_method=model_config["AFRK"].get("method"),  # New
            AFRK_tps_method=model_config["AFRK"].get("tps_method"),  # New
        ).generate()

        LOGGER.info(f"Inference complete")
        LOGGER.info(display_current_time())

def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = fetch_args()

    with open(args.model_config, "rt") as f:
        model_config = yaml.safe_load(f.read())
    with open(args.inference_config, "rt") as f:
        inference_config = yaml.safe_load(f.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 0:
        LOGGER.info(f"Using {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = model_config.get("seed", -1)  # -1 for random seed
    if seed != -1:
        set_seed(seed)
        LOGGER.info(f"Random seed set to {seed} for reproducibility.")

    run_job(model_config, inference_config, device, args.ckpt_iter)
