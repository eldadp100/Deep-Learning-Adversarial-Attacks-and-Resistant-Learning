import os
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from PIL import Image


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
    inputs = []
    targets = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        inputs.append(x)
        targets.append(y)

    inputs = torch.stack(inputs)
    targets = torch.tensor(targets)
    return inputs, targets
