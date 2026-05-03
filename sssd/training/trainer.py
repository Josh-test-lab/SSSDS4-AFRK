import logging
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint  # New
from tqdm import tqdm
import numpy as np  # New

from sssd.core.model_specs import MASK_FN
from sssd.training.utils import training_loss
from sssd.utils.logger import setup_logger
from sssd.utils.utils import find_max_epoch
from sssd.data.utils import get_dataloader  # New
from sssd.utils.utils import find_max_epoch, std_normal  # New
from autoFRK import AutoFRK, to_tensor, garbage_cleaner  # New
from autoFRK.utils.helper import cbrt  # New

LOGGER = setup_logger()


class DiffusionTrainer:
    """
    Train Diffusion Models

    Args:
        dataloader (DataLoader): The training dataloader.
        diffusion_hyperparams (Dict[str, Any]): Hyperparameters for the diffusion process.
        net (nn.Module): The neural network model to be trained.
        device (torch.device): The device to be used for training.
        output_directory (str): Directory to save model checkpoints.
        ckpt_iter (Optional[int, str]): The checkpoint iteration to be loaded; 'max' selects the maximum iteration.
        n_iters (int): Number of iterations to train.
        iters_per_ckpt (int): Number of iterations to save checkpoint.
        iters_per_logging (int): Number of iterations to save training log and compute validation loss.
        learning_rate (float): Learning rate for training.
        only_generate_missing (int): Option to generate missing portions of the signal only.
        masking (str): Type of masking strategy: 'mnr' for Missing Not at Random, 'bm' for Blackout Missing, 'rm' for Random Missing.
        missing_k (int): K missing time steps for each feature across the sample length.
        batch_size (int): Size of each training batch.
        logger (Optional[logging.Logger]): Logger object for logging, defaults to None.
    """

    def __init__(
        self,
        data_path: str,  # New
        diffusion_hyperparams: Dict[str, Any],
        net: nn.Module,
        device: Optional[Union[torch.device, str]],
        output_directory: str,
        ckpt_iter: Union[str, int],
        n_iters: int,
        iters_per_ckpt: int,
        iters_per_logging: int,
        learning_rate: float,
        only_generate_missing: int,
        masking: str,
        missing_k: int,
        batch_size: int,
        enable_spatial_training: bool,  # New
        location_path: str,  # New
        AFRK_method: str,  # New
        AFRK_tps_method: str,  # New
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.device = device
        loader, ts_mean, ts_std, idx_order = get_dataloader(  # New
            path=data_path,
            batch_size=batch_size,
            is_shuffle=True,
            index_order=None,
            device=device,
        )
        #print(f"data index order: {idx_order}")  # New
        self.dataloader = loader
        #self.ts_mean = ts_mean  # New
        #self.ts_std = ts_std    # New
        self.real_data = self.dataloader.dataset.tensors[0].to(self.device) # * self.ts_std + self.ts_mean # New
        loc_loader, _, _, _ = get_dataloader(  # New
            path=location_path,
            batch_size=batch_size,
            is_shuffle=True,
            index_order=idx_order,
            normalize=False,
            device=device,
        )
        self.loc_loader = loc_loader  # New
        self.enable_spatial_training = enable_spatial_training  # New
        self.diffusion_hyperparams = diffusion_hyperparams
        self.net = nn.DataParallel(net).to(device)
        self.output_directory = output_directory
        self.ckpt_iter = ckpt_iter
        self.n_iters = n_iters
        self.iters_per_ckpt = iters_per_ckpt
        self.iters_per_logging = iters_per_logging
        self.learning_rate = learning_rate
        self.only_generate_missing = only_generate_missing
        self.masking = masking
        self.missing_k = missing_k
        self.AFRK_method = AFRK_method  # New
        self.AFRK_tps_method = AFRK_tps_method # New
        self.writer = SummaryWriter(f"{output_directory}/log")
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.logger = logger or LOGGER
        if self.masking not in MASK_FN:
            raise KeyError(f"Please enter a correct masking, but got {self.masking}")

    def _load_checkpoint(self) -> None:
        if self.ckpt_iter == "max":
            self.ckpt_iter = find_max_epoch(self.output_directory)
        if self.ckpt_iter >= 0:
            try:
                model_path = os.path.join(
                    self.output_directory, f"{self.ckpt_iter}.pkl"
                )
                checkpoint = torch.load(model_path, map_location="cpu")

                self.net.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                self.logger.info(
                    f"Successfully loaded model at iteration {self.ckpt_iter}"
                )
            except Exception as e:
                self.ckpt_iter = -1
                self.logger.error(f"No valid checkpoint model found. Error: {e}")
        else:
            self.ckpt_iter = -1
            self.logger.info(
                "No valid checkpoint model found, start training from initialization."
            )

    def _save_model(self, n_iter: int) -> None:
        if n_iter > 0 and n_iter % self.iters_per_ckpt == 0:
            torch.save(
                {
                    "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                os.path.join(self.output_directory, f"{n_iter}.pkl"),
            )

    def _update_mask(self, batch: torch.Tensor) -> torch.Tensor:
        transposed_mask = MASK_FN[self.masking](batch[0], self.missing_k)
        return (
            transposed_mask.permute(1, 0)
            .repeat(batch.size()[0], 1, 1)
            .to(self.device, dtype=torch.float32)
        )

    def _train_per_epoch(self) -> torch.Tensor:
        loss_function=nn.MSELoss()
        # New
        for batch_idx, ((batch,), (loc_batch,)) in enumerate(tqdm(zip(self.dataloader, self.loc_loader), total=min(len(self.dataloader), len(self.loc_loader)), desc=f"{self.n_iter}-th training TS")):
            batch = batch.to(self.device)
            loc_batch = loc_batch.to(self.device)

            mask = self._update_mask(batch)
            loss_mask = ~mask.bool()

            if loss_mask.sum() == 0:
                self.logger.warning(f"Batch {batch_idx} has no valid elements for loss")
                continue

            batch = batch.permute(0, 2, 1)
            assert batch.size() == mask.size() == loss_mask.size()
            
            self.optimizer.zero_grad()
            # loss, choosen_k = training_loss(
            loss = training_loss(
                model=self.net,
                loss_function=loss_function,
                training_data=(batch, batch, mask, loss_mask),
                diffusion_parameters=self.diffusion_hyperparams,
                generate_only_missing=self.only_generate_missing,
                enable_frk= self.enable_spatial_training,
                loc=loc_batch,
                AFRK_method=self.AFRK_method,
                AFRK_tps_method=self.AFRK_tps_method,
                device=self.device,
            )
            loss.backward()
            self.optimizer.step()

            # self.logger.info(f"Iteration:{self.n_iter} choosen_k: {choosen_k}")
            # for k_idx, k_value in enumerate(choosen_k):
            #     self.writer.add_scalar(
            #         tag=f"Train/choosen_k/{k_idx}",
            #         scalar_value=k_value,
            #         global_step=self.n_iter
            #     )

        return loss

    def train(self) -> None:
        self._load_checkpoint()

        n_iter_start = (
            self.ckpt_iter + 2 if self.ckpt_iter == -1 else self.ckpt_iter + 1
        )
        self.logger.info(f"Start the {n_iter_start} iteration")

        for n_iter in range(n_iter_start, self.n_iters + 1):
            self.n_iter = n_iter
            loss = self._train_per_epoch()
            self.writer.add_scalar("Train/Loss", loss.item(), n_iter)
            self.logger.info(f"Iteration: {n_iter} \tLoss: { loss.item()}")
            self._save_model(n_iter)
