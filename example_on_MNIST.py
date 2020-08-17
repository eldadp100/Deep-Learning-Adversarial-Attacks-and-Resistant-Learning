import torch
import torchvision

import attacks
import configs
import datasets
import dls
import helper
import models
import trainer
import os
import shutil
from torchvision.datasets import MNIST

if __name__ == '__main__':
    # Experiment 1: attack network

    # configs
    experiment_configs = configs.TrafficSigns_experiments_configs
    experiment_hps_sets = configs.TrafficSigns_experiments_hps
    show_test_successful_attacks_plots = configs.show_test_successful_attacks_plots
    save_test_successful_attacks_plots = configs.show_test_successful_attacks_plots

    # seed
    if configs.seed is not None:
        torch.manual_seed(configs.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # get datasets
    path_to_save_data = os.path.join(".", "datasets", "mnist_data")
    _training_dataset = MNIST(path_to_save_data, train=True, download=True,
                              transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    _testing_dataset = MNIST(path_to_save_data, train=False, download=True,
                             transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    # create hyperparameters generators
    net_training_hps_gen = helper.GridSearch(experiment_hps_sets["nets_training"])
    fgsm_attack_hps_gen = helper.GridSearch(experiment_hps_sets["FGSM"])
    pgd_attack_hps_gen = helper.GridSearch(experiment_hps_sets["PGD"])

    # loss and general training componenets:
    _loss_fn = experiment_configs["loss_function"]
    stop_criteria = experiment_configs["stopping_criteria"]
    epochs = trainer.Epochs(stop_criteria)


def experiment_1_func(net, _loss_fn, _training_dataset, _testing_dataset, epochs,
                      net_name="", train_attack=None, load_checkpoint=False, save_checkpoint=True):
    if load_checkpoint:
        checkpoint_path = os.path.join(configs.checkpoints_folder, "{}.pt".format(net_name))
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["trained_net"])
        net_hp = checkpoint["net_hp"]
        fgsm_hp = checkpoint["fgsm_hp"]
        pgd_hp = checkpoint["pgd_hp"]
        resistance_results = checkpoint["resistance_results"]

    else:
        # apply hyperparameters-search to get trained network
        net, net_hp, net_acc = helper.full_train_of_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                net_training_hps_gen,
                                                                epochs, device=device, train_attack=train_attack)
        net.eval()

        # attack net (trained) using FGSM:
        _, fgsm_hp, fgsm_score = helper.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                           fgsm_attack_hps_gen, net_hp,
                                                                           attacks.FGSM,
                                                                           device=device,
                                                                           plot_successful_attacks=False)

        # attack net (trained) using PGD:
        _, pgd_hp, pgd_score = helper.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                         pgd_attack_hps_gen, net_hp, attacks.PGD,
                                                                         device=device,
                                                                         plot_successful_attacks=False)

        # measure attacks on test (holdout)
        resistance_results = helper.measure_resistance_on_test(net, _loss_fn, _testing_dataset,
                                                               [(attacks.FGSM, fgsm_hp),
                                                                (attacks.PGD, pgd_hp)],
                                                               plot_successful_attacks=show_test_successful_attacks_plots,
                                                               device=device)

    test_acc = resistance_results["test_acc"]  # the accuracy without applying any attack
    fgsm_res = resistance_results["%fgsm"]
    pgd_res = resistance_results["%pgd"]

    # print scores:
    print("TEST SCORES of {}:".format(net_name))
    print("accuracy on test:            {}".format(test_acc))
    print("%FGSM successful attacks:    {}".format(fgsm_res))
    print("%PGD successful attacks:     {}\n".format(pgd_res))

    # save checkpoint
    res_dict = {
        "trained_net": net,
        "net_hp": net_hp,
        "fgsm_hp": fgsm_hp,
        "pgd_hp": pgd_hp,
        "resistance_results": resistance_results
    }

    if save_checkpoint:
        to_save_res_dict = res_dict
        to_save_res_dict["trained_net"] = net.state_dict()
        checkpoint_path = os.path.join(configs.checkpoints_folder, "{}.pt".format(net_name))
        torch.save(to_save_res_dict, checkpoint_path)

    return res_dict


run_experiment_1 = True
run_experiment_2 = True
run_experiment_3 = True

if __name__ == '__main__' and run_experiment_1:
    # define network
    net_arch = models.MNISTNet().to(device)
    exp1_res_dict = experiment_1_func(net_arch, _loss_fn, _training_dataset, _testing_dataset, epochs,
                                      net_name="Spatial Transformer Network(STN)")
    original_trained_net = exp1_res_dict["trained_net"]
    net_hp = exp1_res_dict["net_hp"]
    fgsm_hp = exp1_res_dict["fgsm_hp"]
    pgd_hp = exp1_res_dict["pgd_hp"]

if __name__ == '__main__' and run_experiment_2:
    # # Experiment 2: build a robust network
    epochs = trainer.Epochs(trainer.ConstantStopping(25))
    # fgsm_robust_net = models.MNISTNet().to(device)
    # fgsm_attack = attacks.FGSM(fgsm_robust_net, _loss_fn, fgsm_hp)
    # experiment_1_func(fgsm_robust_net, _loss_fn, _training_dataset, _testing_dataset, epochs,
    #                   net_name="robust net built using FGSM", train_attack=fgsm_attack)

    epochs.restart()
    pgd_robust_net = models.MNISTNet().to(device)
    pgd_attack = attacks.PGD(pgd_robust_net, _loss_fn, pgd_hp)
    experiment_1_func(pgd_robust_net, _loss_fn, _training_dataset, _testing_dataset, epochs,
                      net_name="robust net built using PGD", train_attack=pgd_attack)

if __name__ == '__main__' and run_experiment_3:
    # Experiment 3: Capacity and robustness
    _loss_fn = experiment_configs["loss_function"]
    stop_criteria = experiment_configs["stopping_criteria"]
    epochs = trainer.Epochs(stop_criteria)

    inc_capacity_nets = []
    base_net_params = {
        "channels_lst": [1, 20, 40],  # the first element is number of input channels
        "extras_blocks_components": [],  # ["dropout"],
        # "p_dropout": 0.1,
        "activation": torch.nn.LeakyReLU,
        "out_size": 10,
        "in_wh": 28,
        "#FC_Layers": 2,
        "CNN_out_channels": 30  # apply 1x1 conv layer to achieve that - to control mem. None to not use.
    }
    for i in range(1, 9):
        base_net_params["channels_lst"] = [1, 10 * i, 20 * i]
        base_net_params["#FC_Layers"] = 1 + i // 2
        base_net_params["CNN_out_channels"] = i * 5
        cap_net = models.create_conv_nn(base_net_params)
        inc_capacity_nets.append(cap_net)
    for i, net in enumerate(inc_capacity_nets):
        net = net.to(device)
        epochs.restart()
        experiment_1_func(net, _loss_fn, _training_dataset, _testing_dataset, epochs,
                          net_name="capacity_{}".format(i))
