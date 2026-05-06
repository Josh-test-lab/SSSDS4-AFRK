import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
import yaml
import time
from autoFRK import MRTS

from sssd.core.model_specs import MODEL_PATH_FORMAT, setup_model
from sssd.data.utils import get_dataloader
from sssd.training.trainer import DiffusionTrainer
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
        "-t",
        "--training_config",
        type=str,
        default="configs/training.yaml",
        help="Training configuration",
    )
    return parser.parse_args()


def setup_output_directory(
    model_config: dict,
    training_config: dict,
) -> str:
    # Build output directory
    local_path = MODEL_PATH_FORMAT.format(
        T=model_config["diffusion"]["T"],
        beta_0=model_config["diffusion"]["beta_0"],
        beta_T=model_config["diffusion"]["beta_T"],
    )
    output_directory = os.path.join(training_config["output_directory"], local_path)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    LOGGER.info("Output directory %s", output_directory)
    return output_directory


def run_job(
    model_config: dict,
    training_config: dict,
    device: Optional[Union[torch.device, str]],
) -> None:
    output_directory = setup_output_directory(model_config, training_config)

    diffusion_hyperparams = calc_diffusion_hyperparams(
        **model_config["diffusion"], device=device
    )

    location_path=os.path.abspath(training_config["location_path"])
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
    net = setup_model(training_config["use_model"], model_config, use_mrts=use_mrts, mrts_dim=klim, device=device)

    if use_mrts:
        LOGGER.info(f"Using MRTS with {klim} dimensions for spatial training.")
    LOGGER.info(display_current_time())
    trainer = DiffusionTrainer(
        data_path=training_config["data"]["train_path"],  # New
        diffusion_hyperparams=diffusion_hyperparams,
        net=net,
        device=device,
        output_directory=output_directory,
        ckpt_iter=training_config.get("ckpt_iter"),
        n_iters=training_config.get("n_iters"),
        iters_per_ckpt=training_config.get("iters_per_ckpt"),
        iters_per_logging=training_config.get("iters_per_logging"),
        learning_rate=training_config.get("learning_rate"),
        only_generate_missing=training_config.get("only_generate_missing"),
        masking=training_config.get("masking"),
        missing_k=training_config.get("missing_k"),
        batch_size=training_config.get("batch_size"),
        use_mrts=use_mrts,  # New
        mrts_dim=klim,  # New
        location_path=location_path,  # New
        logger=LOGGER,
    )
    trainer.train()

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
    with open(args.training_config, "rt") as f:
        training_config = yaml.safe_load(f.read())

    LOGGER.info(f"Model spec: {model_config}")
    LOGGER.info(f"Training spec: {training_config}")

    if torch.cuda.device_count() > 0:
        LOGGER.info(f"Using {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.perf_counter()

    
    seed = model_config.get("seed", -1)  # -1 for random seed
    if seed != -1:
        set_seed(seed)
        LOGGER.info(f"Random seed set to {seed} for reproducibility.")

    run_job(model_config, training_config, device)

    elapsed = time.perf_counter() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    LOGGER.info(f"Elapsed time: {hours}h {minutes}m {seconds:.2f}s")
