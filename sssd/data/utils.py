import random
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sssd.utils.logger import setup_logger
LOGGER = setup_logger()

def merge_all_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in all time points and create rows for missing values.

    Args:
    df (DataFrame): DataFrame containing 'Date', 'Zone', and 'Load' columns.

    Returns:
    DataFrame: A DataFrame with the same columns. The number of rows is hours_df.shape[0] * 11.
    """
    # Create a DataFrame with all hourly time points
    hours_df = pd.DataFrame(
        {"Date": pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="1H")}
    )

    zones = df["Zone"].unique()
    result_all_time = pd.DataFrame()

    for zone in zones:
        # Extract data for the current zone
        load_zone = df.loc[df["Zone"] == zone]

        # Merge with hourly time points
        result = pd.merge(hours_df, load_zone, on="Date", how="left")
        result["Zone"] = zone

        result_all_time = pd.concat([result_all_time, result], axis=0)

    return result_all_time


def load_testing_data(test_data_path: str, num_samples: int) -> torch.Tensor:
    """
    Load and prepare testing data for generation.

    Args:
    - test_data_path (str): Path to the testing data file.
    - num_samples (int): Number of samples per batch.

    Returns:
    - torch.Tensor: Tensor containing the testing data prepared for generation.
    """
    # Load testing data
    testing_data = np.load(test_data_path)

    # Split testing data into batches
    testing_data_batches = np.split(testing_data, testing_data.shape[0] // num_samples)

    # Convert to numpy array and then to torch tensor
    testing_data_tensor = torch.from_numpy(np.array(testing_data_batches)).float()

    # Move tensor to CUDA device if available
    if torch.cuda.is_available():
        testing_data_tensor = testing_data_tensor.cuda()

    return testing_data_tensor


def load_and_split_training_data(
    training_data_load: np.ndarray,
    batch_num: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load and split training data into batches.

    Args:
        training_data_load (np.ndarray): The training data to load and split.
        batch_num (int): The number of batches to create.
        batch_size (int): The size of each batch.
        device (torch.device): The device to move the data to.

    Returns:
        torch.Tensor: The training data split into batches and moved to the specified device.
    """
    total_samples = training_data_load.shape[0]
    if batch_size > total_samples:
        raise ValueError(
            "Batch size exceeds the total number of samples in the training data"
        )

    indices = random.sample(range(total_samples), batch_num * batch_size)
    training_data = training_data_load[indices]
    training_data = np.split(training_data, batch_num, 0)
    training_data = np.array(training_data)
    return torch.from_numpy(training_data).to(device, dtype=torch.float32)


def get_dataloader(
    path: str,
    batch_size: int,
    is_shuffle: bool = True,
    index_order: np.ndarray = None,  # New
    device: Union[str, torch.device] = "cpu",
    num_workers: int = 0,
    normalize: bool = True,  # New
    inference: bool = False,  # New
    missing_k: int = None  # New
) -> DataLoader:
    """
    Get a PyTorch DataLoader for the dataset stored at the given path.

    Args:
        path (str): Path to the dataset file.
        batch_size (int): Size of each batch.
        is_shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        device (Union[str, torch.device], optional): Device to move the data to. Defaults to "cpu".
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 8.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    data = torch.from_numpy(np.load(path)).to(dtype=torch.float32)

    idx_order = None
    if is_shuffle and index_order is None:
        idx_order = torch.randperm(data.shape[0], device=data.device)
        data = data[idx_order]
    elif is_shuffle and index_order is not None:
        idx_order = torch.as_tensor(index_order, dtype=torch.long, device=data.device)
        if idx_order.numel() != data.shape[0]:
            raise ValueError(f"index_order size mismatch: got {idx_order.numel()}, expected {data.shape[0]}")
        if not torch.equal(torch.sort(idx_order).values, torch.arange(data.shape[0], device=data.device)):
            raise ValueError(f"index_order is not a valid permutation of 0..{data.shape[0]-1}")
        data = data[idx_order]
    else:
        idx_order = torch.arange(data.shape[0], device=data.device)

    print(f"Loaded data from {path} with shape {data.shape}")

    if normalize:
        ts_mean = data.mean(dim=1, keepdim=True)
        ts_std = data.std(dim=1, unbiased=False, keepdim=True)
        print(f"Data mean shape: {ts_mean.shape}, std shape: {ts_std.shape}")
        if (ts_std == 0).any():
            LOGGER.error("Standard deviation is zero for one or more sequences; normalization may produce NaN or inf.")
            #ts_std[ts_std == 0] = 1.0
        data = (data - ts_mean) / ts_std
    else:
        ts_mean = torch.zeros_like(data)
        ts_std = torch.ones_like(data)

    if inference and missing_k is not None:
        #zeros = torch.zeros((data.shape[0], missing_k, data.shape[2]), dtype=data.dtype, device=data.device)
        #data = torch.cat([data, zeros], dim=1) 
        repeats = data[:, -1:, :].repeat(1, missing_k, 1)
        #raise ValueError(f"data shape: {data.shape}, repeats shape: {repeats.shape}, repeats value: {repeats}")
        data = torch.cat([data, repeats], dim=1)
    elif inference:
        error_msg = f"In inference mode, missing_k must be specified."
        raise ValueError(error_msg)
    
    dataset = TensorDataset(data)
    pin_memory = device == "cuda" or device == torch.device("cuda")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    print(f"DataLoader created with batch size {batch_size}, and shuffle={is_shuffle}, num_workers={num_workers}, pin_memory={pin_memory}, device={device}, loader length={len(loader)}")

    return loader, ts_mean.to(torch.float32).to(device), ts_std.to(torch.float32).to(device), idx_order


def get_MRTS_dataloader(
    mrts: torch.Tensor,
    batch_size: int,
    is_shuffle: bool = True,
    index_order: np.ndarray = None,  # New
    device: Union[str, torch.device] = "cpu",
    num_workers: int = 0,
    normalize: bool = True,  # New
) -> DataLoader:
    """
    Get a PyTorch DataLoader for the dataset stored at the given path.

    Args:
        mrts (torch.Tensor): The MRTS data.
        batch_size (int): Size of each batch.
        is_shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        device (Union[str, torch.device], optional): Device to move the data to. Defaults to "cpu".
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 8.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    data = mrts

    idx_order = None
    if is_shuffle and index_order is None:
        idx_order = torch.randperm(data.shape[0], device=data.device)
        data = data[idx_order]
    elif is_shuffle and index_order is not None:
        idx_order = torch.as_tensor(index_order, dtype=torch.long, device=data.device)
        if idx_order.numel() != data.shape[0]:
            raise ValueError(f"index_order size mismatch: got {idx_order.numel()}, expected {data.shape[0]}")
        if not torch.equal(torch.sort(idx_order).values, torch.arange(data.shape[0], device=data.device)):
            raise ValueError(f"index_order is not a valid permutation of 0..{data.shape[0]-1}")
        data = data[idx_order]
    else:
        idx_order = torch.arange(data.shape[0], device=data.device)

    if normalize:
        ts_mean = data.mean(dim=1, keepdim=True)
        ts_std = data.std(dim=1, unbiased=False, keepdim=True)
        print(f"Data mean shape: {ts_mean.shape}, std shape: {ts_std.shape}")
        if (ts_std == 0).any():
            LOGGER.error("Standard deviation is zero for one or more sequences; normalization may produce NaN or inf.")
            #ts_std[ts_std == 0] = 1.0
        data = (data - ts_mean) / ts_std
    else:
        ts_mean = torch.zeros_like(data)
        ts_std = torch.ones_like(data)
    
    dataset = TensorDataset(data)
    pin_memory = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    print(f"DataLoader created with batch size {batch_size}, and shuffle={is_shuffle}, num_workers={num_workers}, pin_memory={pin_memory}, device={device}, loader length={len(loader)}")

    return loader, ts_mean.to(torch.float32).to(device), ts_std.to(torch.float32).to(device), idx_order
