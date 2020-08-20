import time
import torch
import torchvision
from torchvision.datasets import MNIST
import attacks
import configs
import datasets
import dls
import helper
import models
import trainer
import os
import shutil
import logger

"""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wolf/sagieb/course/miniconda3/lib/

export CUDA_VISIBLE_DEVICES=7
conda activate hw4_env
cd SignsTrafficTest
python example.py 
"""

if __name__ == '__main__':
    # Experiment 1: attack network

    # configs
    experiment_configs = configs.MNIST_experiments_configs
    experiment_hps_sets = configs.MNIST_experiments_hps
    experiment_results_folder = os.path.join(configs.results_folder, "MNIST")
    experiment_checkpoints_folder = os.path.join(configs.checkpoints_folder, "MNIST")
    logger_path = os.path.join(experiment_results_folder, "log.txt")
    plots_folder = os.path.join(experiment_results_folder, "plots")

    # paths existence validation and initialization
    if not os.path.exists(configs.results_folder):
        os.mkdir(configs.results_folder)
    if not os.path.exists(experiment_results_folder):
        os.mkdir(experiment_results_folder)
    if not os.path.exists(configs.checkpoints_folder):
        os.mkdir(experiment_results_folder)
    if not os.path.exists(experiment_checkpoints_folder):
        os.mkdir(experiment_checkpoints_folder)
    if os.path.exists(plots_folder):
        shutil.rmtree(plots_folder)
        time.sleep(.0001)
    os.mkdir(plots_folder)
    if os.path.exists(logger_path):
        os.remove(logger_path)

    # seed
    if configs.seed is not None:
        # np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)

    # set logger
    logger.init_log(logger_path)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log_print("execution device: {}".format(device))

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
    training_stop_criteria = experiment_configs["training_stopping_criteria"]
    adv_training_stop_criteria = experiment_configs["adversarial_training_stopping_criteria"]
    epochs = trainer.Epochs(training_stop_criteria)  # epochs obj for not adversarial training
    adv_epochs = trainer.Epochs(adv_training_stop_criteria)  # epochs obj for adversarial training


def experiment_1_func(net, _loss_fn, _training_dataset, _testing_dataset, epochs, net_name="", train_attack=None,
                      load_checkpoint=False, save_checkpoint=False, show_plots=False, save_plots=False):
    """
    :param net:
    :param _loss_fn:
    :param _training_dataset:
    :param _testing_dataset:
    :param epochs:
    :param net_name:
    :param train_attack:
    :param load_checkpoint:
    :param save_checkpoint:
    :return:
    """
    if load_checkpoint:
        checkpoint_path = os.path.join(experiment_checkpoints_folder, "{}.pt".format(net_name))
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["trained_net"])
        net_hp = checkpoint["net_hp"]
        fgsm_hp = checkpoint["fgsm_hp"]
        pgd_hp = checkpoint["pgd_hp"]
        resistance_results = checkpoint["resistance_results"]

    else:
        # apply hyperparameters-search to get a trained network
        net_state_dict, net_hp = helper.full_train_of_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                  net_training_hps_gen, epochs, device=device,
                                                                  train_attack=train_attack)
        net.load_state_dict(net_state_dict)
        net.eval()  # from now on we only evaluate net.

        # attack selected net using FGSM:
        fgsm_hp, fgsm_score = helper.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                        fgsm_attack_hps_gen, net_hp, attacks.FGSM,
                                                                        device=device, plot_results=False,
                                                                        save_figs=False, figs_path=plots_folder)

        # attack selected net using PGD:
        pgd_hp, pgd_score = helper.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                      pgd_attack_hps_gen, net_hp, attacks.PGD,
                                                                      device=device, plot_results=False,
                                                                      save_figs=False,
                                                                      figs_path=plots_folder)

        # measure attacks on test (holdout)
        resistance_results = helper.measure_resistance_on_test(net, _loss_fn, _testing_dataset,
                                                               to_attacks=[(attacks.FGSM, fgsm_hp),
                                                                           (attacks.PGD, pgd_hp)],
                                                               device=device,
                                                               plot_results=show_plots,
                                                               save_figs=save_plots,
                                                               figs_path=plots_folder,
                                                               plots_title=net_name)

    # unpack resistance_results
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

    if save_checkpoint and not load_checkpoint:
        to_save_res_dict = res_dict
        to_save_res_dict["trained_net"] = net.state_dict()
        checkpoint_path = os.path.join(experiment_checkpoints_folder, "{}.pt".format(net_name))
        torch.save(to_save_res_dict, checkpoint_path)

    return res_dict


run_experiment_1 = True
run_experiment_2 = True
run_experiment_3 = False

if __name__ == '__main__' and run_experiment_1:
    # define network
    original_net = models.MNISTNet().to(device)
    exp1_res_dict = experiment_1_func(original_net, _loss_fn, _training_dataset, _testing_dataset, epochs,
                                      net_name="STN (Spatial Transformer Network)",
                                      save_checkpoint=configs.save_checkpoints,
                                      load_checkpoint=configs.load_checkpoints,
                                      show_plots=configs.show_attacks_plots,
                                      save_plots=configs.save_attacks_plots)


# Build robust network with PGD and FGSM and compare them
def experiment_2_func(exp1_res_dict, net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                      net_name="", load_checkpoint=False, save_checkpoint=False, show_plots=False, save_plots=False):
    net_hp = exp1_res_dict["net_hp"]
    fgsm_hp = exp1_res_dict["fgsm_hp"]
    pgd_hp = exp1_res_dict["pgd_hp"]
    pgd_res, fgsm_res = None, None

    # adversarial_epochs.restart()
    # net.apply(helper.weight_reset)
    # fgsm_robust_net = net.to(device)
    # fgsm_attack = attacks.FGSM(fgsm_robust_net, _loss_fn, fgsm_hp)
    # fgsm_res = experiment_1_func(fgsm_robust_net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
    #                   net_name="{} with FGSM adversarial training".format(net_name), train_attack=fgsm_attack,
    #                   load_checkpoint=load_checkpoint, save_checkpoint=save_checkpoint, show_plots=show_plots,
    #                   save_plots=save_plots)

    adversarial_epochs.restart()
    net.apply(helper.weight_reset)
    pgd_robust_net = net.to(device)
    pgd_attack = attacks.PGD(pgd_robust_net, _loss_fn, pgd_hp)
    pgd_res = experiment_1_func(pgd_robust_net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                                net_name="{} with PGD adversarial training".format(net_name), train_attack=pgd_attack,
                                load_checkpoint=load_checkpoint, save_checkpoint=save_checkpoint, show_plots=show_plots,
                                save_plots=save_plots)

    return fgsm_res, pgd_res


# Experiment 2: Build robust networks + Compare PGD and FGSM adversarial trainings
if __name__ == '__main__' and run_experiment_2:
    experiment_2_func(exp1_res_dict, models.MNISTNet(), _loss_fn, _training_dataset, _testing_dataset, adv_epochs,
                      net_name="STN (Spatial Transformer Network)",
                      save_checkpoint=configs.save_checkpoints,
                      load_checkpoint=configs.load_checkpoints,
                      show_plots=configs.show_attacks_plots,
                      save_plots=configs.save_attacks_plots)

if __name__ == '__main__' and run_experiment_3:
    # Experiment 3: Capacity and robustness
    inc_capacity_nets = []
    base_net_params = {
        "extras_blocks_components": [],  # ["dropout"],
        # "p_dropout": 0.1,
        "activation": torch.nn.LeakyReLU,
        "out_size": 10,
        "in_wh": 28
    }

    for i in range(1, 10):
        if i == 1:
            base_net_params["channels_lst"] = [1, 1]
            base_net_params["#FC_Layers"] = 1
            base_net_params["CNN_out_channels"] = None
        if 2 <= i <= 4:
            base_net_params["channels_lst"] = [1, 1 * i, 2 * i]
            base_net_params["#FC_Layers"] = 1
            base_net_params["CNN_out_channels"] = None
        if 5 <= i <= 8:
            base_net_params["channels_lst"] = [1, 3 * i, 6 * i]
            base_net_params["#FC_Layers"] = 2
            base_net_params["CNN_out_channels"] = 4 * i
        if i == 9:
            base_net_params["channels_lst"] = [1, 100, 200, 70]
            base_net_params["#FC_Layers"] = 4
            base_net_params["CNN_out_channels"] = 40

        cap_net = models.create_conv_nn(base_net_params)
        inc_capacity_nets.append(cap_net)

    # run experiment 1 and 2 on each capacity
    for i, net in enumerate(inc_capacity_nets):
        net = net.to(device)
        epochs.restart()
        exp1 = experiment_1_func(net, _loss_fn, _training_dataset, _testing_dataset, epochs,
                                 net_name="capacity_{}".format(i))
        experiment_2_func(exp1, net, _loss_fn, _training_dataset, _testing_dataset, adv_epochs,
                          net_name="capacity_{}".format(i))
