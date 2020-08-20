import torch
from torch.optim import Adam, SGD
from torch import nn
import trainer
from torchvision import transforms
import os

# paths
data_root_dir = os.path.join(".", "data")
checkpoints_folder = os.path.join(".", "checkpoints")
results_folder = os.path.join(".", "results_folder")

# general configurations:
save_checkpoints = True
load_checkpoints = False  # To use a saved checkpoint instead re-training.
show_attacks_plots = False  # plots cannot be displayed in NOVA
save_attacks_plots = True
seed = None  # Specify Random Seed. Helps to debug issues that appear seldom.
dls_num_workers = 1  # Dataloaders number of workers - 0 for loading using the main process
imgs_to_show = 4  # maximal number images to show in a grid of images

"""
The default split ratios in the proj. suppose we have N samples then:
%train = train_ratio * val_ratio
%val = train_ratio * (1-val_ratio)
%test = 1-train_ratio
"""
train_ratio = 0.8
val_ratio = 0.7

# experiment specific configs and hyperparameters:
TrafficSigns_experiments_configs = {
    "data_transform": transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ]),
    "hps_construction_method": "grid",  # grid / random only.
    "adversarial_training_stopping_criteria": trainer.ConstantStopping(7),
    "training_stopping_criteria": trainer.ConstantStopping(2),
    # "stopping_criteria": trainer.ConstantStopping(5),  # trainer.TimerStopping(10),  # trainer.ConstantStopping(5),
    "loss_function": nn.CrossEntropyLoss(),  # the nets architectures are built based on CE loss
}

MNIST_experiments_configs = {
    "hps_construction_method": "grid",  # grid / random only.
    "adversarial_training_stopping_criteria": trainer.ConstantStopping(5),
    "training_stopping_criteria": trainer.ConstantStopping(1),
    "loss_function": nn.CrossEntropyLoss(),  # the nets architectures are built based on CE loss
}

TrafficSigns_experiments_hps = {
    "FGSM": {
        "epsilon": [0.007, 0.001, 0.1, 0.3],  # [0.01, 0.001, 0.0001],
    },

    "PGD": {
        "alpha": [0.01],  # [0.01, 0.001, 0.0001],
        "steps": [40],
        "epsilon": [0.3]  # [5, 20, 40, 100]
    },

    "nets_training": {
        "lr": [0.00005, 0.0001],  # [0.001, 0.0005, 0.01],
        "batch_size": [64],
        # "optimizer": [torch.optim.SGD, torch.optim.Adam]
    },
}

MNIST_experiments_hps = TrafficSigns_experiments_hps
