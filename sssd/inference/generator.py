import logging
import os
from typing import Dict, Iterable, Optional, Union

import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm  # New
import csv  # New

from sssd.core.model_specs import MASK_FN
from sssd.data.utils import get_dataloader  # New
from sssd.utils.logger import setup_logger
from sssd.utils.utils import find_max_epoch, sampling
from autoFRK import AutoFRK, to_tensor, garbage_cleaner  # New

LOGGER = setup_logger()


class DiffusionGenerator:
    """
    Generate data based on ground truth.

    Args:
        net (torch.nn.Module): The neural network model.
        device (Optional[Union[torch.device, str]]): The device to run the model on (e.g., 'cuda' or 'cpu').
        diffusion_hyperparams (dict): Dictionary of diffusion hyperparameters.
        local_path (str): Local path format for the model.
        testing_data (torch.Tensor): Tensor containing testing data.
        output_directory (str): Path to save generated samples.
        batch_size (int): Number of samples to generate.
        ckpt_path (str): Checkpoint directory.
        ckpt_iter (str): Pretrained checkpoint to load; 'max' selects the maximum iteration.
        masking (str): Type of masking: 'mnr' (missing not at random), 'bm' (black-out), 'rm' (random missing).
        missing_k (int): Number of missing time points for each channel across the length.
        only_generate_missing (int): Whether to generate only missing portions of the signal:
                                      - 0 (all sample diffusion),
                                      - 1 (generate missing portions only).
        saved_data_names (Iterable[str], optional): Names of data arrays to save (default is ("imputation", "original", "mask")).
        logger (Optional[logging.Logger], optional): Logger object for logging messages (default is None).
    """

    def __init__(
        self,
        net: torch.nn.Module,
        device: Optional[Union[torch.device, str]],
        diffusion_hyperparams: dict,
        local_path: str,
        data_path: str,  # New
        output_directory: str,
        batch_size: int,
        ckpt_path: str,
        ckpt_iter: str,
        masking: str,
        missing_k: int,
        only_generate_missing: int,
        enable_spatial_prediction: bool,  # New
        enable_spatial_normalization: bool,  # New
        known_location_path: str,  # New
        unknown_location_path: str,  # New
        AFRK_method: str,  # New
        AFRK_tps_method: str,  # New
        saved_data_names: Iterable[str] = ("imputation", "original", "mask"),
        logger: Optional[logging.Logger] = None,
    ):
        self.net = net
        self.device = device
        self.diffusion_hyperparams = diffusion_hyperparams
        self.local_path = local_path
        loader, ts_mean, ts_std, idx_order = get_dataloader(  # New
            path=data_path,
            batch_size=batch_size,
            device=device,
            is_shuffle=False,
            index_order=None,
            inference=True,
            missing_k=missing_k
        )
        self.dataloader = loader
        self.ts_mean = ts_mean  # New
        self.ts_std = ts_std    # New
        self.batch_size = batch_size
        self.masking = masking
        self.missing_k = missing_k
        self.only_generate_missing = only_generate_missing
        self.enable_spatial_prediction = enable_spatial_prediction  # New
        self.enable_spatial_normalization = enable_spatial_normalization  # New
        self.known_location_path = known_location_path  # New
        self.unknown_location_path = unknown_location_path  # New
        self.missing_k = missing_k  # New
        self.AFRK_method = AFRK_method  # New
        self.AFRK_tps_method = AFRK_tps_method # New
        self.logger = logger or LOGGER
        self.output_directory = self._prepare_output_directory(
            output_directory, local_path, ckpt_iter
        )
        self.saved_data_names = saved_data_names
        self._load_checkpoint(ckpt_path, ckpt_iter)

    def _load_checkpoint(self, ckpt_path: str, ckpt_iter: str) -> None:
        """Load a checkpoint for the given neural network model."""
        ckpt_path = os.path.join(ckpt_path, self.local_path)
        if ckpt_iter == "max":
            ckpt_iter = find_max_epoch(ckpt_path)
        model_path = os.path.join(ckpt_path, f"{ckpt_iter}.pkl")
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            self.net.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(f"Successfully loaded model at iteration {ckpt_iter}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found at {model_path}") from e
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")

    def _prepare_output_directory(
        self, output_directory: str, local_path: str, ckpt_iter: str
    ) -> str:
        """Prepare the output directory to save generated samples."""
        ckpt_iter_str = (
            "max"
            if ckpt_iter == "max"
            else f"imputation_multiple_{int(ckpt_iter) // 1000}k"
        )
        output_directory = os.path.join(output_directory, local_path, ckpt_iter_str)
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o775)
        self.logger.info(f"Output directory: {output_directory}")
        return output_directory

    def _update_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """Update mask based on the given batch."""
        transposed_mask = MASK_FN[self.masking](batch[0], self.missing_k)
        return (
            transposed_mask.permute(1, 0)
            .repeat(batch.size()[0], 1, 1)
            .to(self.device, dtype=torch.float32)
        )

    def _save_data(
        self,
        results: Dict[str, np.ndarray],
        index: int,
    ) -> None:
        """Save generated_series, batch, and mask data arrays."""

        for name, data in results.items():
            if name in self.saved_data_names:
                filename = f"{name}{index}.npy"
                np.save(os.path.join(self.output_directory, filename), data)
        
        if "choosen_k" in results:
            save_path = os.path.join(self.output_directory, "choosen_k.csv")
            with open(save_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["choosen_k"])
                for k in results["choosen_k"]:
                    writer.writerow([k])

    def _autoFRK_generate(
        self,
        sssd_inference,
        with_known_loc: bool = True
    ) -> torch.Tensor:
        LOGGER.info(f"Start autoFRK inference step")
        dtype = sssd_inference.dtype
        device = sssd_inference.device
        sssd_inference = to_tensor(sssd_inference, dtype = dtype, device = device)
        N, T, V = sssd_inference.shape
        known_loc = to_tensor(np.load(self.known_location_path), dtype=dtype, device=device)
        unknown_loc = to_tensor(np.load(self.unknown_location_path), dtype=dtype, device=device)
        if known_loc.ndim == 1:
            known_loc = known_loc.view(-1, 1)
        if unknown_loc.ndim == 1:
            unknown_loc = unknown_loc.view(-1, 1)
        missing_loc = unknown_loc.shape[0]
        autoFRK_inference = torch.zeros((missing_loc, T, V), dtype=dtype, device=device)
        frk = AutoFRK(
            logger_level=30,
            dtype=dtype,
            device=device,
        )

        mrts = None
        choosen_k = []
        for variable in tqdm(range(V), desc=f"inferencing autoFRK"):
            data_slice = sssd_inference[:, :, variable]
            try:
                _ = frk.forward(
                    data=data_slice,
                    loc=known_loc,
                    G = mrts,
                    method=self.AFRK_method,
                    tps_method=self.AFRK_tps_method,
                    requires_grad=False
                )
                pred = frk.predict(
                    newloc = unknown_loc
                )['pred.value']
                mrts = frk.obj['G'] if mrts is None else mrts
                choosen_k.append(frk.obj['G']['MRTS'].shape[1])

            except torch._C._LinAlgError:
                LOGGER.warning(f"Skipped variable={variable} due to ill-conditioned matrix")
                pred = torch.zeros((missing_loc, T), dtype=dtype, device=device)
            
            autoFRK_inference[:, :, variable] = pred

        if with_known_loc:
            autoFRK_inference = torch.cat([sssd_inference, autoFRK_inference], dim=0)

        return autoFRK_inference, choosen_k
    
    def generate(self) -> list:
        """Generate samples using the given neural network model."""
        all_generated = []
        for index, (batch,) in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="inferencing sssd"):
            batch = batch.to(self.device)
            mask = self._update_mask(batch)
            batch = batch.permute(0, 2, 1)

            generated_series = (
                sampling(
                    net=self.net,
                    size=batch.shape,
                    diffusion_hyperparams=self.diffusion_hyperparams,
                    cond=batch,
                    mask=mask,
                    only_generate_missing=self.only_generate_missing,
                    device=self.device,
                )
            )

            all_generated.append(generated_series)
        sssd_inference = torch.cat(all_generated, dim=0).permute(0, 2, 1)
        ts_mean = to_tensor(self.ts_mean, dtype = sssd_inference.dtype, device = sssd_inference.device)
        ts_std = to_tensor(self.ts_std, dtype = sssd_inference.dtype, device = sssd_inference.device)
        sssd_inference = sssd_inference * ts_std + ts_mean

        if self.enable_spatial_prediction:

            # new feature for this chunk
            if self.enable_spatial_normalization:
                sp_mean = sssd_inference.mean(dim=0, keepdim=True)
                sp_std = sssd_inference.std(dim=0, unbiased=False, keepdim=True)
                sssd_inference = (sssd_inference - sp_mean) / (sp_std + 1e-8)

            autoFRK_inference, choosen_k = self._autoFRK_generate(
                sssd_inference = sssd_inference,
                with_known_loc = True
            )

            # new feature for this chunk
            if self.enable_spatial_normalization:
                autoFRK_inference = autoFRK_inference * sp_std + sp_mean

            results = {'imputation': autoFRK_inference.detach().cpu().numpy(), 'choosen_k': choosen_k}
        else:
            results = {'imputation': sssd_inference.detach().cpu().numpy()}
        self._save_data(results, 0)
