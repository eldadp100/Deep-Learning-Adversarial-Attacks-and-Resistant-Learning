import os
import torch
import torchvision
from torchvision.datasets import MNIST
from torch import nn
import numpy as np
import attacks
import dls
import helper
import trainer
import configs


def initialize():
    if configs.seed is not None:
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)


path_to_save_data = os.path.join(".", "datasets", "mnist_data")
_train_dataset = MNIST(path_to_save_data, train=True, download=True,
                       transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
_test_dataset = MNIST(path_to_save_data, train=False, download=True,
                      transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

_loss_fn = nn.CrossEntropyLoss()

experiment_configs = configs.MINIST_experiments_configs

device = 'cpu'


# TODO: change method name and make more general to next experiments uses
def experiment_1(net, loss_fn, train_dataset, test_dataset, network_name=""):
    """
    :param net:
    :param loss_fn:
    :param train_dataset:
    :param test_dataset:
    :param network_name:
    :param plot_successful_attacks:
    :return:
    """

    # load hyperparameters
    experiment_configs = configs.MINIST_experiments_configs
    experiment_hps_dict = configs.MINIST_experiments_hps

    net_train_hps_gen = helper.GridSearch(experiment_hps_dict["nets_training"])
    pgd_attack_hps_gen = helper.GridSearch(experiment_hps_dict["PGD"])
    fgsm_attack_hps_gen = helper.GridSearch(experiment_hps_dict["PGD"])

    epochs = trainer.Epochs(stopping_criteria=experiment_configs["stopping_criteria"])

    # train nets
    net, net_best_hp, net_best_acc = helper.full_train_of_nn_with_hps(net, loss_fn, train_dataset, net_train_hps_gen,
                                                                      epochs, device=device)
    print(net_best_hp)

    # Attack with FGSM:
    _, best_fgsm_hp, best_fgsm_score = helper.full_attack_of_trained_nn_with_hps(net, loss_fn, train_dataset,
                                                                                 fgsm_attack_hps_gen,
                                                                                 net_best_hp, attacks.FGSM,
                                                                                 device=device,
                                                                                 plot_successful_attacks=False)
    print(best_fgsm_score)

    # Attack with PGD:
    _, best_pgd_hp, best_pgd_score = helper.full_attack_of_trained_nn_with_hps(net, loss_fn, train_dataset,
                                                                               pgd_attack_hps_gen, net_best_hp,
                                                                               attacks.PGD,
                                                                               device=device,
                                                                               plot_successful_attacks=False)
    print(best_pgd_score)

    # measure on test (holdout)!!!
    resistance_results = helper.measure_resistance_on_test(net, loss_fn, test_dataset,
                                                           [(attacks.FGSM, best_fgsm_hp),
                                                            (attacks.PGD, best_pgd_hp)],
                                                           plot_successful_attacks=True)

    original_acc = resistance_results["original"]
    fgsm_test_acc = resistance_results["fgsm"]
    pgd_test_acc = resistance_results["pgd"]

    # print scores:
    print("TEST SCORES of {}:".format(network_name))
    print("ORIGINAL accuracy:      {}".format(original_acc))
    print("FGSM accuracy:          {}".format(fgsm_test_acc))
    print("PGD accuracy:           {}\n".format(pgd_test_acc))

    ret_dict = {
        "net_best_hp": net_best_hp,
        "best_fgsm_hp": best_fgsm_hp,
        "best_pgd_hp": best_pgd_hp,

        "original_acc": original_acc,
        "fgsm_test_acc": fgsm_test_acc,
        "pgd_test_acc": pgd_test_acc,

    }
    return ret_dict


# set the network
mnist_net_params = {
    "channels_lst": [1, 10, 30],  # the first element is number of input channels
    "extras_blocks_components": ["dropout"],
    "p_dropout": 0.1,
    "activation": nn.ReLU,
    "out_size": 10,
    "in_wh": 28,
    "#FC_Layers": 2
}
mnist_net = helper.create_conv_nn(mnist_net_params)
mnist_net.to(device)
exp_1_results = experiment_1(mnist_net, _loss_fn, _train_dataset, _test_dataset, network_name="MNIST experiment_1")

nn_training_hp = exp_1_results["net_best_hp"]
fgsm_hp = exp_1_results["best_fgsm_hp"]
pgd_hp = exp_1_results["best_pgd_hp"]

train_dl, val_dl = dls.get_train_val_dls(_train_dataset, nn_training_hp["batch_size"])
epochs = trainer.Epochs(stopping_criteria=experiment_configs["stopping_criteria"])

fgsm_resistant_mnist_net = helper.create_conv_nn(mnist_net_params)
fgsm_nn_optimizer = torch.optim.SGD(fgsm_resistant_mnist_net.parameters(), nn_training_hp["lr"])
fgsm_attack = attacks.FGSM(fgsm_resistant_mnist_net, _loss_fn, fgsm_hp)
trainer.train_nn(fgsm_resistant_mnist_net, fgsm_nn_optimizer, _loss_fn, train_dl, epochs, fgsm_attack)

pgd_resistant_mnist_net = helper.create_conv_nn(mnist_net_params)
pgd_nn_optimizer = torch.optim.SGD(pgd_resistant_mnist_net.parameters(), nn_training_hp["lr"])
pgd_attack = attacks.PGD(pgd_resistant_mnist_net, _loss_fn, pgd_hp)
trainer.train_nn(pgd_resistant_mnist_net, pgd_nn_optimizer, _loss_fn, train_dl, epochs, pgd_attack)

# measure resistance on test:
fgsm_resistance_results = helper.measure_resistance_on_test(fgsm_resistant_mnist_net, _loss_fn, _test_dataset,
                                                            [(attacks.FGSM, fgsm_hp),
                                                             (attacks.PGD, pgd_hp)],
                                                            plots_title="robust net built using FGSM")
pgd_resistance_results = helper.measure_resistance_on_test(pgd_resistant_mnist_net, _loss_fn, _test_dataset,
                                                           [(attacks.FGSM, fgsm_hp),
                                                            (attacks.PGD, pgd_hp)],
                                                           plots_title="robust net built using PGD")
# print_results:
print("using FGSM:")
print(fgsm_resistance_results)
print("PGD:")
print(pgd_resistance_results)

inc_capacity_nets = []
base_net_params = {
    "channels_lst": [1, 20, 40],  # the first element is number of input channels
    "extras_blocks_components": [],  # ["dropout"],
    # "p_dropout": 0.1,
    "activation": nn.LeakyReLU,
    "out_size": 10,
    "in_wh": 28,
    "#FC_Layers": 2
}
for i in range(8):
    base_net_params["channels_lst"] = [1, 2 ** i, 2 ** (i + 1)]
    base_net_params["#FC_Layers"] = 1 + i // 4
    cap_net = helper.create_conv_nn(base_net_params)
    inc_capacity_nets.append(cap_net)

for i, net in enumerate(inc_capacity_nets):
    experiment_1(net, _loss_fn, _train_dataset, _test_dataset, network_name="capacity_{}".format(i))
