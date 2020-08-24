from torch import nn
import trainer
from torchvision import transforms
import os

# paths
data_root_dir = os.path.join(".", "data")
checkpoints_folder = os.path.join(".", "checkpoints")
results_folder = os.path.join(".", "results_folder")

# general configurations:
save_checkpoints = False
load_checkpoints = False  # To use a saved checkpoint instead re-training.
show_attacks_plots = False  # plots cannot be displayed in NOVA
save_attacks_plots = True
show_validation_accuracy_each_epoch = True  # becomes True if using early stopping
seed = None  # Specify Random Seed. Helps to debug issues that appear seldom.
dls_num_workers = 1  # Dataloaders number of workers - 0 for loading using the main process
imgs_to_show = 4  # maximal number images to show in a grid of images
val_ratio = 0.7

TrafficSigns_experiments_configs = {
    "data_transform": transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))  # To get [0,1] range
    ]),
    "hps_construction_method": "grid",  # grid only
    "adversarial_training_stopping_criteria": trainer.ConstantStopping(10),
    "training_stopping_criteria": trainer.ConstantStopping(4),
    "loss_function": nn.CrossEntropyLoss(),  # the nets architectures are built based on CE loss
}

TrafficSigns_experiments_hps = {
    "FGSM_attack": {
        "epsilon": [0.3],  # 2 / 255,
    },

    "PGD_attack": {
        "alpha": [0.01],
        "steps": [30, 20, 7],
        "epsilon": [0.3]  # 2 / 255,
    },

    "FGSM_train": {
        "epsilon": [0.05],
    },

    "PGD_train": {
        "alpha": [0.01],
        "steps": [30],
        "epsilon": [0.05]
    },

    "nets_training": {
        "lr": [0.0005],
        "batch_size": [64],
    },
}

MNIST_experiments_configs = {
    "hps_construction_method": "grid",  # grid / random only.
    "adversarial_training_stopping_criteria": trainer.ConstantStopping(8),
    "training_stopping_criteria": trainer.ConstantStopping(5),
    "loss_function": nn.CrossEntropyLoss(),  # the nets architectures are built based on CE loss
}

MNIST_experiments_hps = {
    "FGSM_attack": {
        "epsilon": [0.3],
    },

    "PGD_attack": {
        "alpha": [0.01],
        "steps": [40, 100, 20, 7],
        "epsilon": [0.3]
    },

    "FGSM_train": {
        "epsilon": [0.3],
    },

    "PGD_train": {
        "alpha": [0.01],
        "steps": [100],
        "epsilon": [0.3]
    },

    "nets_training": {
        "lr": [0.001],
        "batch_size": [128],
    },
}
