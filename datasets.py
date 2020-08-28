import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import configs
import numpy as np


class GTSRB(Dataset):
    """ Taken from https://github.com/tomlawrenceuk/GTSRB-Dataloader"""
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None):
        self.root_dir = root_dir
        self.sub_directory = 'trainingset' if train else 'testset'  # TODO: RENAME
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(root_dir, self.base_folder, self.sub_directory, self.csv_file_name)  # paths file
        self.csv_data = pd.read_csv(csv_file_path)
        self.transform = transform

        self.memorize_samples = [None] * len(self.csv_data)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        # memorize:
        if self.memorize_samples[idx] is not None:
            return self.memorize_samples[idx]
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory, self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)
        img_label = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        self.memorize_samples[idx] = (img, img_label)
        return img, img_label


def get_data_labels(dataset: Dataset):
    """ split dataset into tensor of data inputs and thier labels"""
    inputs = []
    targets = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        inputs.append(x)
        targets.append(y)

    inputs = torch.stack(inputs)
    targets = torch.tensor(targets)
    return inputs, targets


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
