import torch
import attacks
import configs
import datasets
import dls
import helper
import models
import trainer

if __name__ == '__main__':
    # Experiment 1: attack network

    # configs
    experiment_configs = configs.TrafficSigns_experiments_configs
    experiment_hps_sets = configs.TrafficSigns_experiments_hps
    if configs.seed is not None:
        torch.manual_seed(configs.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = experiment_configs["data_transform"]

    # get datasets
    _training_dataset = datasets.GTSRB(root_dir='./data', train=True, transform=transform)
    _testing_dataset = datasets.GTSRB(root_dir='./data', train=False, transform=transform)

    # create hyperparameters generators
    net_training_hps_gen = helper.GridSearch(experiment_hps_sets["nets_training"])
    fgsm_attack_hps_gen = helper.GridSearch(experiment_hps_sets["FGSM"])
    pgd_attack_hps_gen = helper.GridSearch(experiment_hps_sets["PGD"])

    # define network and training components
    net = models.TrafficSignNet().to(device)
    _loss_fn = experiment_configs["loss_function"]
    stop_criteria = experiment_configs["stopping_criteria"]
    epochs = trainer.Epochs(stop_criteria)

    # apply hyperparameters-search to get trained network
    trained_net, net_hp, net_acc = helper.full_train_of_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                    net_training_hps_gen, epochs, device=device)

    # attack trained_net using FGSM:
    _, fgsm_hp, fgsm_score = helper.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                       fgsm_attack_hps_gen, net_hp, attacks.FGSM,
                                                                       device=device, plot_successful_attacks=False)

    # attack trained_net using PGD:
    _, pgd_hp, pgd_score = helper.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                     pgd_attack_hps_gen, net_hp, attacks.PGD,
                                                                     device=device, plot_successful_attacks=False)

    # measure attacks on test (holdout)!!!
    resistance_results = helper.measure_resistance_on_test(net, _loss_fn, _testing_dataset,
                                                           [(attacks.FGSM, fgsm_hp),
                                                            (attacks.PGD, pgd_hp)],
                                                           plot_successful_attacks=True,
                                                           device=device)

    test_acc = resistance_results["test_acc"]  # the accuracy without applying any attack
    fgsm_res = resistance_results["%fgsm"]
    pgd_res = resistance_results["%pgd"]

    # print scores:
    print("TEST SCORES of {}:".format("Spatial Transformer Network (STN)"))
    print("accuracy on test:            {}".format(test_acc))
    print("%FGSM successful attacks:    {}".format(fgsm_res))
    print("%PGD successful attacks:     {}\n".format(pgd_res))

    # Experiment 2: build a robust network
    train_dl, val_dl = dls.get_train_val_dls(_training_dataset, net_hp["batch_size"])
    epochs.restart()  # TODO: add get_instance() method in stop_criteria class

    fgsm_robust_net = models.TrafficSignNet().to(device)
    fgsm_nn_optimizer = torch.optim.SGD(fgsm_robust_net.parameters(), net_hp["lr"])  # TODO: change to ADAM
    fgsm_attack = attacks.FGSM(fgsm_robust_net, _loss_fn, fgsm_hp)
    trainer.train_nn(fgsm_robust_net, fgsm_nn_optimizer, _loss_fn, train_dl, epochs, fgsm_attack)

    pgd_robust_net = models.TrafficSignNet().to(device)
    pgd_nn_optimizer = torch.optim.SGD(pgd_robust_net.parameters(), net_hp["lr"])  # TODO: change to ADAM
    pgd_attack = attacks.PGD(pgd_robust_net, _loss_fn, pgd_hp)
    trainer.train_nn(pgd_robust_net, pgd_nn_optimizer, _loss_fn, train_dl, epochs, pgd_attack)

    # measure resistance on test:
    fgsm_resistance_results = helper.measure_resistance_on_test(fgsm_robust_net, _loss_fn, _testing_dataset,
                                                                [(attacks.FGSM, fgsm_hp),
                                                                 (attacks.PGD, pgd_hp)],
                                                                plot_successful_attacks=True,
                                                                plots_title="robust net built using FGSM",
                                                                device=device)
    print("FGSM based trained robust net attacking results:")
    print(fgsm_resistance_results)

    pgd_resistance_results = helper.measure_resistance_on_test(pgd_robust_net, _loss_fn, _testing_dataset,
                                                               [(attacks.FGSM, fgsm_hp),
                                                                (attacks.PGD, pgd_hp)],
                                                               plot_successful_attacks=True,
                                                               plots_title="robust net built using PGD",
                                                               device=device)
    print("PGD based trained robust net attacking results:")
    print(pgd_resistance_results)
