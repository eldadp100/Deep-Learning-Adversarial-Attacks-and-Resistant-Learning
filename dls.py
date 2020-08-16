"""
    Data loaders file.
    every not standard dataset that is in use is implemented here.
"""
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset
import configs
import numpy as np


#
# def get_train_val_test_dls(dataset: Dataset, batch_size):
#     """
#     :param dataset
#     :param hp: contains hyper-parameters like batch_size, random seed
#     :return: train & validation & test data loaders of type DataLoader.
#     """
#
#     ds_size = len(dataset)
#     indices = list(range(ds_size))
#     np.shuffle(indices)
#     test_split_idx = int(ds_size * configs.train_ratio)
#     val_split_idx = int(test_split_idx * configs.val_ratio)
#     train_indices, val_indices, test_indices = indices[:val_split_idx], \
#                                                indices[val_split_idx:test_split_idx], indices[test_split_idx:]
#
#     train_sampler = SubsetRandomSampler(train_indices)
#     val_sampler = SubsetRandomSampler(val_indices)
#     test_sampler = SubsetRandomSampler(test_indices)
#
#     train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
#     validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
#     test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
#     return train_loader, validation_loader, test_loader
#

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

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader


def dataset_to_dataloader(dataset: Dataset, batch_size):
    """ A function that gets dataset and returns simple data loader.
        It is useful when the dataset comes slitted (to train and test)"""

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def split_to_data_labels(subset_dataset: Subset):
    """
    :param subset_dataset: Subset type from torch utils.
    :return:
    """
    indices = subset_dataset.indices
    xs = []
    ys = []
    for i in indices:
        x, y = subset_dataset[i]
        xs.append(x)
        ys.append(y)

    return torch.tensor(xs), torch.tensor(ys)
