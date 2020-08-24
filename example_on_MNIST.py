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

export CUDA_VISIBLE_DEVICES=2
conda activate hw4_env
cd SignsTrafficTest
python example.py 

jupyter notebook --no-browser --port=4999
"""

run_experiment_1 = False
run_experiment_2 = True
run_experiment_3 = False

if __name__ == '__main__':
    # Experiment 1: attack network

    # configs
    experiment_configs = configs.MNIST_experiments_configs
    experiment_hps_sets = configs.MNIST_experiments_hps
    experiment_results_folder = os.path.join(configs.results_folder, "MNIST")
    experiment_checkpoints_folder = os.path.join(configs.checkpoints_folder, "MNIST")
    logger_path = os.path.join(experiment_results_folder, "mnist_experiment_4_capacity_log.txt")
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
    fgsm_attack_hps_gen = helper.GridSearch(experiment_hps_sets["FGSM_attack"])
    pgd_attack_hps_gen = helper.GridSearch(experiment_hps_sets["PGD_attack"])
    fgsm_training_hps_gen = helper.GridSearch(experiment_hps_sets["FGSM_train"])
    pgd_training_hps_gen = helper.GridSearch(experiment_hps_sets["PGD_train"])

    # loss and general training componenets:
    _loss_fn = experiment_configs["loss_function"]
    training_stop_criteria = experiment_configs["training_stopping_criteria"]
    adv_training_stop_criteria = experiment_configs["adversarial_training_stopping_criteria"]
    epochs = trainer.Epochs(training_stop_criteria)  # epochs obj for not adversarial training
    adv_epochs = trainer.Epochs(adv_training_stop_criteria)  # epochs obj for adversarial training


def experiment_1_func(net, _loss_fn, _training_dataset, _testing_dataset, epochs, net_name="", train_attack=None,
                      attack_training_hps_gen=None, load_checkpoint=False, save_checkpoint=False, show_plots=False,
                      save_plots=False, show_validation_accuracy_each_epoch=False):
    """
    This experiment applies training or adversarial training on the given network. Then it prints the resistant
    measurements on both PGD and FGSM attacks.
    :param net: the given network. we initialize net parameters.
    :param _loss_fn: loss function.
    :param _training_dataset: the dataset to train on.
    :param _testing_dataset: a separate dataset to measure resistance and accuracy on.
    :param epochs: Epochs object that manages the training procedure. see Epochs class in trainer.py for more details.
    :param net_name: the network name is used in plotting titles and checkpoints files names.
    :param train_attack: in case we want to apply an adversarial training instead of normal training.
    :param load_checkpoint: use pre-trained model. To use that verify the existence of one in checkpoints folder.
    :param save_checkpoint: save the trained model.
    :return: the resistance results + found hps in the hyperparameter searches + trained network.
    """
    if load_checkpoint:
        checkpoint_path = os.path.join(experiment_checkpoints_folder, "{}.pt".format(net_name))
        logger.log_print("load network from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["trained_net"])
        net_hp = checkpoint["net_hp"]
        fgsm_hp = checkpoint["fgsm_hp"]
        pgd_hp = checkpoint["pgd_hp"]
        resistance_results = checkpoint["resistance_results"]

    else:
        hps_gen = net_training_hps_gen
        if train_attack is not None:
            hps_gen = helper.concat_hps_gens(net_training_hps_gen, attack_training_hps_gen)

        # apply hyperparameters-search to get a trained network
        net_state_dict, net_hp = helper.full_train_of_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                  hps_gen, epochs, device=device,
                                                                  train_attack=train_attack,
                                                                  show_validation=show_validation_accuracy_each_epoch)
        net.load_state_dict(net_state_dict)
        net.eval()  # from now on we only evaluate net.

        logger.log_print("training selected hyperparams: {}".format(str(net_hp)))

        # attack selected net using FGSM:
        fgsm_hp, fgsm_score = helper.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                        fgsm_attack_hps_gen, net_hp, attacks.FGSM,
                                                                        device=device, plot_results=False,
                                                                        save_figs=False, figs_path=plots_folder)
        logger.log_print("FGSM attack selected hyperparams: {}".format(str(fgsm_hp)))

        # attack selected net using PGD:
        pgd_hp, pgd_score = helper.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                      pgd_attack_hps_gen, net_hp, attacks.PGD,
                                                                      device=device, plot_results=False,
                                                                      save_figs=False,
                                                                      figs_path=plots_folder)
        logger.log_print("PGD attack selected hyperparams: {}".format(str(pgd_hp)))

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
    logger.log_print("TEST SCORES of {}:".format(net_name))
    logger.log_print("accuracy on test:                         {}".format(test_acc))
    logger.log_print("accuracy on FGSM constructed examples:    {}".format(fgsm_res))
    logger.log_print("accuracy on PGD constructed examples:     {}".format(pgd_res))

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
        logger.log_print("save network to {}".format(checkpoint_path))
        torch.save(to_save_res_dict, checkpoint_path)

    return res_dict


if __name__ == '__main__' and run_experiment_1:
    net_name = "CNN-MNIST"
    logger.log_print("\n")
    logger.log_print("Experiment 1 on {}".format(net_name))
    # define network
    original_net = models.MNISTNet().to(device)
    logger.log_print("Network architecture")
    logger.log_print(str(original_net))
    exp1_res_dict = experiment_1_func(original_net, _loss_fn, _training_dataset, _testing_dataset, epochs,
                                      net_name=net_name,
                                      save_checkpoint=configs.save_checkpoints,
                                      load_checkpoint=configs.load_checkpoints,
                                      show_plots=configs.show_attacks_plots,
                                      save_plots=configs.save_attacks_plots,
                                      show_validation_accuracy_each_epoch=configs.show_validation_accuracy_each_epoch)


# Build robust network with PGD and FGSM and compare them
def experiment_2_func(net_arch, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                      net_name="", load_checkpoint=False, save_checkpoint=False, show_plots=False, save_plots=False,
                      show_validation_accuracy_each_epoch=False):
    adversarial_epochs.restart()
    fgsm_robust_net = net_arch().to(device)
    experiment_1_func(fgsm_robust_net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                      net_name="{} with FGSM adversarial training".format(net_name), train_attack=attacks.FGSM,
                      attack_training_hps_gen=fgsm_training_hps_gen,
                      load_checkpoint=load_checkpoint, save_checkpoint=save_checkpoint, show_plots=show_plots,
                      save_plots=save_plots, show_validation_accuracy_each_epoch=show_validation_accuracy_each_epoch)

    adversarial_epochs.restart()
    pgd_robust_net = net_arch().to(device)
    experiment_1_func(pgd_robust_net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                      net_name="{} with PGD adversarial training".format(net_name), train_attack=attacks.PGD,
                      attack_training_hps_gen=pgd_training_hps_gen,
                      load_checkpoint=load_checkpoint, save_checkpoint=save_checkpoint, show_plots=show_plots,
                      save_plots=save_plots, show_validation_accuracy_each_epoch=show_validation_accuracy_each_epoch)


# Experiment 2: Build robust networks + Compare PGD and FGSM adversarial trainings
if __name__ == '__main__' and run_experiment_2:
    logger.log_print("\n")
    net_name = "A"
    logger.log_print("Experiment 2 on {}".format(net_name))
    experiment_2_func(models.MNISTNet, _loss_fn, _training_dataset, _testing_dataset,
                      adv_epochs,
                      net_name=net_name,
                      save_checkpoint=configs.save_checkpoints,
                      load_checkpoint=configs.load_checkpoints,
                      show_plots=configs.show_attacks_plots,
                      save_plots=configs.save_attacks_plots,
                      show_validation_accuracy_each_epoch=configs.show_validation_accuracy_each_epoch)

if __name__ == '__main__' and run_experiment_3:
    # Experiment 3: Capacity and robustness
    inc_capacity_nets = []
    base_net_params = {
        "extras_blocks_components": [],  # ["dropout"],
        "p_dropout": 0.1,
        "activation": torch.nn.LeakyReLU,
        "out_size": 10,
        "in_wh": 28
    }

    for i in range(1, 9):
        if 1 <= i <= 4:
            base_net_params["channels_lst"] = [1, 2 ** i, 3 ** i]
            base_net_params["#FC_Layers"] = 2
            base_net_params["CNN_out_channels"] = 10 * i
        if 5 <= i <= 8:
            base_net_params["channels_lst"] = [1, 2 ** (i // 2), 2 ** i, 2 ** i]
            base_net_params["#FC_Layers"] = 4
            base_net_params["CNN_out_channels"] = 80

        cap_net = models.create_conv_nn(base_net_params)
        inc_capacity_nets.append(cap_net)

    # run experiment 1 and 2 on each capacity
    for i, net in enumerate(inc_capacity_nets):
        net = net.to(device)
        epochs.restart()
        experiment_1_func(net, _loss_fn, _training_dataset, _testing_dataset, epochs,
                          net_name="capacity_{}".format(i + 1))
        # experiment_2_func(net, _loss_fn, _training_dataset, _testing_dataset, adv_epochs,
        #                   net_name="capacity_{}".format(i))
