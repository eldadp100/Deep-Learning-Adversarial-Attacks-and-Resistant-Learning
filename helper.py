import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import dls
import trainer


def show_img_lst(imgs, titles=None, x_labels=None, main_title=None, columns=2, plot_img=True, save_img=True,
                 save_path=None):
    """

    :param imgs:
    :param titles:
    :param x_labels:
    :param main_title:
    :param columns:
    :param plot_img:
    :param save_img:
    :param save_path:
    :return:
    """
    if save_img:
        assert save_path is not None

    # plot images
    img_shape = imgs[0].shape
    fig = plt.figure()
    rows = np.ceil(len(imgs) / columns)
    for i in range(len(imgs)):
        fig.add_subplot(rows, columns, i + 1)
        img = imgs[i].detach()
        # fix img
        if img_shape[0] == 1:
            img = img[0]
        else:
            img = np.transpose(img, (1, 2, 0))
        # plot img
        plt.imshow(img)
        if titles is not None:
            plt.xlabel(titles[i])
        if x_labels is not None:
            plt.xlabel(x_labels[i])

    if main_title is not None:
        fig.suptitle(main_title)

    # plot / save
    if plot_img:
        plt.show()
    if save_img:
        plt.savefig(save_path)
    plt.clf()


# Hyperparameter Generation:
class HyperparamsGen:
    """ Abstract class for hyperparameters generation techniques. """

    def __init__(self, hps_dict):
        self.hps_dict = hps_dict
        self.size_num = None

    def next(self):
        """ returns NONE if there are no more hps """
        pass

    def restart(self):
        pass

    def size(self):
        """
            number of possible hyperparams:
            c = 1
            for key in self.hps_dict.keys():
                c *= len(self.hps_dict[key])
            return c
        """
        if self.size_num is None:
            self.size_num = np.prod([len(self.hps_dict[k]) for k in self.hps_dict.keys()])
        return self.size_num


class GridSearch(HyperparamsGen):
    """
        Goes over all possible combinations of hps (hyperparameters).
        Implemented as a generator to save memory - crucial when there are many hps.
    """

    def __init__(self, hps_dict):
        super().__init__(hps_dict)
        self.hps_keys = list(hps_dict.keys())
        self.values_size = [len(hps_dict[k]) - 1 for k in self.hps_keys]
        self.indices = [0] * len(self.hps_keys)

    def next(self):
        """ returns NONE if there are no more hps"""
        if self.indices[0] > self.values_size[0]:
            return None

        # construct HP:
        hp = {}
        for idx, val_idx in enumerate(self.indices):
            key = self.hps_keys[idx]
            hp[key] = self.hps_dict[key][val_idx]

        # next hp indices
        i = len(self.indices) - 1
        while i >= 0 and self.indices[i] == self.values_size[i]:
            self.indices[i] = 0
            i -= 1
        self.indices[max(0, i)] += 1

        return hp

    def restart(self):
        self.indices = [0] * len(self.hps_keys)


def measure_resistance_on_test(net, loss_fn, test_dataset, to_attacks, device=None, plots_title="", plot_results=False,
                               save_figs=False, figs_path=None):
    """
    :param net:
    :param loss_fn:
    :param test_dataset:
    :param to_attacks:
    :param plot_results: plot successful attacks
    :param plots_title:
    :param device:
    :param save_figs:
    :param figs_path:
    :return:
    """
    results = {}
    test_dataloader = DataLoader(test_dataset, batch_size=100)
    original_acc = trainer.measure_classification_accuracy(net, test_dataloader, device=device)
    for attack_class, attack_hp in to_attacks:
        attack = attack_class(net, loss_fn, attack_hp)
        title = "resistance {}: {}".format(attack.name, plots_title)
        test_score = attack.test_attack(test_dataloader,
                                        main_title=title,
                                        plot_results=plot_results,
                                        save_results_figs=save_figs,
                                        figs_path=os.path.join(figs_path, title),
                                        device=device)
        results["%{}".format(attack.name)] = test_score

    results["test_acc"] = original_acc
    return results


def reset_net_parameters(model):
    """ taken from: https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034/2?u=eldadperetz"""
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def full_train_of_nn_with_hps(net, loss_fn, train_dataset, hps_gen, epochs, device=None, train_attack=None):
    """
    cross validation hyperparameter search.
    :param net:
    :param loss_fn:
    :param train_dataset:
    :param hps_gen:
    :param epochs:
    :param device:
    :param train_attack:
    :return: net, net_best_hp, net_best_acc. net is trained on full train dataset (not splitted)
    """
    hps_gen.restart()
    if hps_gen.size() > 1:
        net_best_hp, net_best_acc = None, 0
        while True:
            hp = hps_gen.next()
            if hp is None:
                break

            # restart previous execution
            reset_net_parameters(net)
            epochs.restart()

            # set train and val dataloaders, optimizer
            train_dl, val_dl = dls.get_train_val_dls(train_dataset, hp["batch_size"])
            nn_optimizer = torch.optim.Adam(net.parameters(), hp["lr"])

            # train network:
            trainer.train_nn(net, nn_optimizer, loss_fn, train_dl, epochs, device=device, attack=train_attack)

            # measure on validation set:
            net_acc = trainer.measure_classification_accuracy(net, val_dl, device=device)
            if net_acc >= net_best_acc:
                net_best_acc = net_acc
                net_best_hp = hp
    else:
        net_best_hp = hps_gen.next()
        net_best_acc = None

    full_train_dl = DataLoader(train_dataset, batch_size=net_best_hp["batch_size"], shuffle=True)
    # reset_net_parameters(net)
    epochs.restart()
    nn_optimizer = torch.optim.Adam(net.parameters(), net_best_hp["lr"])
    trainer.train_nn(net, nn_optimizer, loss_fn, full_train_dl, epochs, device=device, attack=train_attack)
    return net, net_best_hp, net_best_acc


def full_attack_of_trained_nn_with_hps(net, loss_fn, train_dataset, hps_gen, selected_nn_hp, attack_method, device=None,
                                       plots_title="", plot_results=False, save_figs=False, figs_path=None):
    """
    hyperparameter search in order to find the hp with highest attack score (i.e. prob to successfully attack).
    :param net:
    :param loss_fn:
    :param train_dataset:
    :param hps_gen:
    :param selected_nn_hp:
    :param attack_method:
    :param device:
    :param plots_title:
    :param plot_results:
    :param save_figs:
    :param figs_path:
    :return: best_hp, best_score (approximately prob to successfully attack)
    """
    hps_gen.restart()
    train_dl, val_dl = dls.get_train_val_dls(train_dataset, selected_nn_hp["batch_size"])

    best_hp, best_score = None, 1.0
    while True:
        hp = hps_gen.next()
        if hp is None:
            break

        attack = attack_method(net, loss_fn, hp)
        title = "resistance {}: {}".format(attack.name, plots_title)
        score = attack.test_attack(
            val_dl,
            main_title=title,
            plot_results=plot_results,
            save_results_figs=save_figs,
            figs_path=os.path.join(figs_path, title),
            device=device
        )
        if score <= best_score:  # lower score is better
            best_score = score
            best_hp = hp

    return best_hp, best_score
