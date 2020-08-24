"""
    Data loaders file.
    every not standard dataset that is in use is implemented here.
"""
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
import configs
import numpy as np


def get_train_val_dls(dataset: Dataset, batch_size):
    """
    same as previous function without test (use this function in case that test is already separated).
    all inputs as in prev function (i.e. get_train_val_test_dls).
    :return: train & val data loaders of type DataLoader.
    """

    ds_size = len(dataset)
    indices = list(range(ds_size))
    np.random.shuffle(indices)
    val_split_idx = int(ds_size * configs.val_ratio)
    train_indices, val_indices = indices[:val_split_idx], indices[val_split_idx:]

    # train_subset = Subset(dataset, train_indices)
    # val_subset = Subset(dataset, val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))

    return train_loader, validation_loader


def dataset_to_dataloader(dataset: Dataset, batch_size):
    """ A function that gets dataset and returns simple data loader.
        It is useful when the dataset comes slitted (to train and test)"""

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

