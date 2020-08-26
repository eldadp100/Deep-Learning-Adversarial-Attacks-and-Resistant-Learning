import torch
from torch import nn
import matplotlib.pyplot as plt
import configs
import os

experiment_configs = configs.TrafficSigns_experiments_configs
experiment_hps_sets = configs.TrafficSigns_experiments_hps
experiment_results_folder = os.path.join(configs.results_folder, "MNIST")
experiment_checkpoints_folder = os.path.join(configs.checkpoints_folder, "MNIST")

logger_path = os.path.join(experiment_results_folder, "inspection_log.txt")
plots_folder = os.path.join(experiment_results_folder, "plots")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = os.path.join(experiment_checkpoints_folder,
                               "{}.pt".format("STN (Spatial Transformer Network) with PGD adversarial training"))
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint["trained_net"]
first_layer = state_dict['cnn.0.weight']
second_layer = state_dict['cnn.2.weight']
second_layer_bias = state_dict['cnn.2.bias']
print(first_layer.shape)
print(second_layer.shape)
uti_lst = []
for i in range(first_layer.shape[0]):
    utilization = 0
    for j in range(second_layer.shape[0]):
        utilization += torch.norm(second_layer[j][i]).item()
        utilization -= second_layer_bias[j].item()
    uti_lst.append(utilization)

import numpy as np
for i in np.argsort(uti_lst)[-10:]:
    print(uti_lst[i])
    print(first_layer[i])
    if first_layer[i].shape[0] > 1:
        plt.imshow(np.transpose(first_layer[i].detach(), (1, 2, 0)))
    else:
        plt.imshow(first_layer[i][0])

    plt.show()
    plt.clf()
