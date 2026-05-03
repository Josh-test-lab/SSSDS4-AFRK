from typing import Dict, Tuple

import torch

from sssd.utils.utils import std_normal
from sssd.utils.logger import setup_logger  # New
LOGGER = setup_logger()  # New
from autoFRK import AutoFRK  # New

def training_loss(
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    training_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    diffusion_parameters: Dict[str, torch.Tensor],
    generate_only_missing: int = 1,
    enable_frk: bool = True,
    loc: torch.Tensor = None,
    AFRK_method: str = "fast",
    AFRK_tps_method: str = "rectangular",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the training loss of epsilon and epsilon_theta.

    Args:
        model (torch.nn.Module): The neural network model.
        loss_function (torch.nn.Module): The loss function, default is nn.MSELoss().
        training_data (tuple): Training data tuple containing (time_series, condition, mask, loss_mask).
        diffusion_parameters (dict): Dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams.
                                     Note, the tensors need to be cuda tensors.
        generate_only_missing (int): Flag to indicate whether to only generate missing values (default=1).
        device (str): Device to run the computations on (default="cuda").

    Returns:
        torch.Tensor: Training loss.
    """

    # Unpack diffusion hyperparameters
    T, alpha_bar = diffusion_parameters["T"], diffusion_parameters["Alpha_bar"]

    # Unpack training data
    time_series, condition, mask, loss_mask = training_data

    batch_size = time_series.shape[0]

    # Sample random diffusion steps for each batch element
    diffusion_steps = torch.randint(T, size=(batch_size, 1, 1)).to(device)

    # debug
    if torch.isnan(diffusion_steps).any():
        LOGGER.warning("diffusion_steps contains NaN")
    
    # Generate Gaussian noise, applying mask if specified
    noise = (
        time_series * mask.float()
        + std_normal(time_series.shape, device) * (1 - mask).float()
        if generate_only_missing
        else std_normal(time_series.shape, device)
    )

    # debug
    if torch.isnan(noise).any():
        LOGGER.warning("noise contains NaN")
        LOGGER.info(f"noise stats: min={noise.min().item()}, max={noise.max().item()}, mean={noise.mean().item()}")

    # Compute x_t from q(x_t|x_0)
    transformed_series = (
        torch.sqrt(alpha_bar[diffusion_steps]) * time_series
        + torch.sqrt(1 - alpha_bar[diffusion_steps]) * noise
    )

    # debug
    if torch.isnan(transformed_series).any():
        LOGGER.warning("transformed_series contains NaN")
        LOGGER.info(f"transformed_series stats: min={transformed_series.min().item()}, max={transformed_series.max().item()}, mean={transformed_series.mean().item()}")

    # New: Integrate AutoFRK for spatial prediction
    frk_pred = None
    if enable_frk and loc is not None:
        temp = transformed_series.permute(1, 0, 2)
        frk_pred = torch.zeros_like(transformed_series)
        for i in range(temp.shape[0]):
            success = False
            while not success:
                try:
                    df = temp[i]
                    frk_model = AutoFRK(
                        device=df.device,
                        dtype=df.dtype,
                        logger_level=30
                        )
                    frk_model.forward(
                        data=df,
                        loc=loc,
                        method=AFRK_method,
                        tps_method=AFRK_tps_method,
                        requires_grad=True
                        )
                    pred_res = frk_model.predict(newloc=loc)
                    frk_pred[:, i, :] = pred_res['pred.value']
                    success = True  # successful and exit loop
                except Exception as e:
                    LOGGER.warning(f"Failed to process record {i}. Retrying. Error: {e}")

    # Predict epsilon according to epsilon_theta
    epsilon_theta = model(
        (transformed_series, condition, mask, diffusion_steps.view(batch_size, 1), frk_pred)
    )

    # debug
    if torch.isnan(epsilon_theta).any():
        LOGGER.warning("epsilon_theta contains NaN")
        LOGGER.info(f"epsilon_theta stats: min={epsilon_theta.min().item()}, max={epsilon_theta.max().item()}, mean={epsilon_theta.mean().item()}")

    # Compute loss
    if generate_only_missing:
        loss = loss_function(epsilon_theta[loss_mask], noise[loss_mask])
    else:
        loss = loss_function(epsilon_theta, noise)

    # debug
    if torch.isnan(loss):
        LOGGER.warning("loss contains NaN")
        LOGGER.info(f"loss value: {loss.item()}")

    return loss # , choosen_k
