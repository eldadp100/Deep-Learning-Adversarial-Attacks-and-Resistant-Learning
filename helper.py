import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import configs
import datasets
import dls
import trainer


def show_img_lst(imgs, titles=None, x_labels=None, main_title=None, columns=2):
    img_shape = imgs[0].shape
    fig = plt.figure()
    rows = np.ceil(len(imgs) / columns)
    for i in range(len(imgs)):
        fig.add_subplot(rows, columns, i + 1)
        img = imgs[i].detach()
        if img_shape[0] == 1:
            img = img[0]
        else:
            img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        if titles is not None:
            plt.xlabel(titles[i])
        if x_labels is not None:
            plt.xlabel(x_labels[i])

    if main_title is not None:
        plt.title(main_title)
    plt.show()


# Hyperparameter Generation:
class HyperparamsGen:
    def __init__(self, hps_dict):
        self.hps_dict = hps_dict

    def next(self):
        """ returns NONE if there are no more hps"""
        pass


class GridSearch(HyperparamsGen):
    """ Implemented as a generator for a case of many hyperparams"""

    def __init__(self, hps_dict):
        super().__init__(hps_dict)
        self.hps_dict = hps_dict
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


def measure_resistance_on_test(net, loss_fn, test_dataset, to_attacks, plot_successful_attacks=False, plots_title="",
                               device=None):
    """
    :param net:
    :param loss_fn:
    :param test_dataset:
    :param to_attacks:
    :param plot_successful_attacks:
    :param plots_title:
    :param device:
    :return:
    """
    results = {}
    test_dataloader = DataLoader(test_dataset, batch_size=100)
    original_acc = trainer.measure_classification_accuracy(net, test_dataloader, device=device)
    for attack_class, attack_hp in to_attacks:
        attack = attack_class(net, loss_fn, attack_hp)
        test_score, _ = attack.test_attack(test_dataloader,
                                           main_title="{} + {}".format(attack.name, plots_title),
                                           plot_successful_attacks=plot_successful_attacks,
                                           device=device)
        results["%{}".format(attack.name)] = test_score

    results["test_acc"] = original_acc
    return results


def reset_net_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def full_train_of_nn_with_hps(net, loss_fn, train_dataset, hps_gen, epochs, device=None):
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
        trainer.train_nn(net, nn_optimizer, loss_fn, train_dl, epochs, device=device)

        # measure on validation set:
        net_acc = trainer.measure_classification_accuracy(net, val_dl, device=device)
        if net_acc >= net_best_acc:
            net_best_acc = net_acc
            net_best_hp = hp

    full_train_dl = DataLoader(train_dataset, batch_size=net_best_hp["batch_size"], shuffle=True)
    net.apply(weight_reset)
    epochs.restart()
    nn_optimizer = torch.optim.Adam(net.parameters(), net_best_hp["lr"])
    trainer.train_nn(net, nn_optimizer, loss_fn, full_train_dl, epochs, device=device)
    return net, net_best_hp, net_best_acc


def full_attack_of_trained_nn_with_hps(net, loss_fn, train_dataset, hps_gen, selected_nn_hp, attack_method,
                                       device=None, plot_successful_attacks=False):
    train_dl, val_dl = dls.get_train_val_dls(train_dataset, selected_nn_hp["batch_size"])

    best_attack, best_hp, best_score = None, None, 1
    while True:
        hp = hps_gen.next()
        if hp is None:
            break

        attack = attack_method(net, loss_fn, hp)
        score, _ = attack.test_attack(val_dl, plot_successful_attacks=plot_successful_attacks, device=device)
        if score <= best_score:  # lower score is better
            best_score = score
            best_hp = hp
            best_attack = attack

    return best_attack, best_hp, best_score


# test
if __name__ == '__main__':
    imgs = (np.random.rand(10 * 64 * 64 * 3) * 256).reshape((10, 64, 64, 3)).astype(int)
    show_img_lst(imgs)
