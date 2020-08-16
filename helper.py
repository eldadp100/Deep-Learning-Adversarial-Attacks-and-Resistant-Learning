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


# Conv NN constructors:
class ConvNN(nn.Module):
    def __init__(self, params):
        """
        The purpose in using this class in the way it built is to make the process of creating CNNs with the ability
        to control its capacity in efficient way (programming efficiency - not time efficiency).
        I found it very useful in constructing experiments. I tried to make this class general as possible.
        :param params: a dictionary with the following attributes:
            capacity influence:
            - channels_lst: lst of the channels sizes. the network complexity is inflected mostly by this parameter
              * for efficiency channels_lst[0] is the number of input channels
            - #FC_Layers: number of fully connected layers
            - extras_blocks_components: in case we want to add layers from the list ["dropout", "max_pool", "batch norm"]
                                        to each block we can do it. Their parameters are attributes of this dict also.
              * notice that if max_pool in extras_blocks_components then we reduce dims using max_pool instead conv
                layer (the conv layer will be with stride 1 and padding)
            - p_dropout: the dropout parameter

            net structure:
            - in_wh: input width and height
        """
        super().__init__()
        self.params = params
        channels_lst = params["channels_lst"]
        extras_block_components = params["extras_blocks_components"]

        assert 2 <= len(channels_lst) <= 5
        conv_layers = []
        for i in range(1, len(channels_lst)):
            if i is not None:
                """
                Dims calculations: next #channels x (nh-filter_size/2)+1 x (nw-filter_size/2)+1
                """
                filter_size, stride = (4, 2) if "max_pool" not in extras_block_components else (3, 1)
                conv_layers.append(nn.Conv2d(channels_lst[i - 1], channels_lst[i], filter_size, stride, 1, bias=False))

                for comp in extras_block_components:
                    if comp == "dropout":
                        conv_layers.append(nn.Dropout(params["p_dropout"]))
                    if comp == "max_pool":
                        conv_layers.append(nn.MaxPool2d(2, 2))
                    if comp == "batch_norm":
                        conv_layers.append(nn.BatchNorm2d(channels_lst[i]))

                conv_layers.append(params["activation"]())
        self.cnn = nn.Sequential(*conv_layers)

        lin_layers = []
        wh = params["in_wh"] // (2 ** (len(channels_lst) - 1))  # width and height of last layer output
        lin_layer_width = channels_lst[-1] * (wh ** 2)
        for _ in range(params["#FC_Layers"] - 1):
            lin_layers.append(nn.Linear(lin_layer_width, lin_layer_width))
        lin_layers.append(nn.Linear(lin_layer_width, params["out_size"]))
        self.linear_nn = nn.Sequential(*lin_layers)

        """ we use CE loss so we don't need to apply softmax (for test loss we also use the same CE. for accuracy
            we choose the highest value - this property is saved under softmax)"""

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(1, *x.shape)
        assert x.shape[2] == x.shape[3] == self.params["in_wh"]
        assert x.shape[1] == self.params["channels_lst"][0]

        cnn_output = self.cnn(x).view((x.shape[0], -1))
        lin_output = self.linear_nn(cnn_output)
        return lin_output


def create_conv_nn(params):
    return ConvNN(params)


# TODO: here use dataloader transformers...
def fix_data_labels(xs, ys, sample_shape):
    if xs[0].shape != sample_shape:
        # xs = xs.view(xs.shape[0], dataset[0][0].shape[0], xs.shape[1], xs.shape[2])
        xs = xs.view(xs.shape[0], 1, xs.shape[1], xs.shape[2])
    xs = xs.type(torch.FloatTensor)
    return xs, ys


# TODO: CHANGE IT - use the dataloader used transformers to do that automatically...
def subset_dataset_split(dataset: Subset):
    xs, ys = dataset.dataset.data[dataset.indices], dataset.dataset.targets[dataset.indices]
    return fix_data_labels(xs, ys, dataset[0][0].shape)


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
                                           plot_successful_attacks=plot_successful_attacks)
        results["%{}".format(attack.name)] = test_score

    results["test_acc"] = original_acc
    return results


def full_train_of_nn_with_hps(net, loss_fn, train_dataset, hps_gen, epochs, device=None):
    best_net, net_best_hp, net_best_acc = None, None, 0

    while True:
        hp = hps_gen.next()
        if hp is None:
            break

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
            best_net = net

    return best_net, net_best_hp, net_best_acc


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
