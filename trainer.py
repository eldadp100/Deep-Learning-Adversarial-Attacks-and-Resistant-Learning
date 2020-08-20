import time
import torch
from torch.utils.data import DataLoader
import logger


class StoppingCriteria:
    def __init__(self):
        self.epoch_num = 0
        self.epochs_summaries = list()

    def update(self, epochs_summaries):
        """UPDATE AFTER EVERY EPOCH"""
        self.epoch_num += 1
        self.epochs_summaries = epochs_summaries  # by pointer - faster then append

    def stop(self):
        pass

    def restart(self):
        pass


class EarlyStopping(StoppingCriteria):
    """ To use this epochs method a val_dl must be specified """

    def __init__(self, max_epochs_num):
        """
        :param max_epochs_num: number of epochs to stop after
        """
        super().__init__()
        self.max_epochs_num = max_epochs_num

    def stop(self):
        if self.epochs_summaries[-1]["val_acc"] is not None and self.epoch_num > 1:
            improved_con = self.epochs_summaries[-2]["val_acc"] < self.epochs_summaries[-1]["val_acc"]
        else:
            improved_con = True
        max_epochs_con = True if self.epoch_num >= self.max_epochs_num else False
        return improved_con and max_epochs_con

    def restart(self):
        self.epoch_num = 0


class ConstantStopping(StoppingCriteria):
    def __init__(self, max_epochs_num):
        """
        :param max_epochs_num: number of epochs to stop after
        """
        super().__init__()
        self.max_epochs_num = max_epochs_num

    def stop(self):
        return True if self.epoch_num >= self.max_epochs_num else False

    def restart(self):
        self.epoch_num = 0


class TimerStopping(StoppingCriteria):
    def __init__(self, max_time):
        """
        :param max_time: number of seconds to stop after. It will be with a delay of at most epoch training time.
        """
        super().__init__()
        self.max_time = max_time
        self.start_time = time.time()

    def stop(self):
        curr_time = time.time()
        return True if curr_time - self.start_time > self.max_time else False

    def restart(self):
        self.start_time = time.time()


class Epochs:
    def __init__(self, stopping_criteria: StoppingCriteria):
        self.epoch_number = 0
        self.epochs_summaries = list()
        self.stopping_criteria = stopping_criteria
        self.latest_net_params = [None, None]

    def update(self, epoch_summary, state_dict):
        self.epoch_number += 1
        self.epochs_summaries.append(epoch_summary)
        self.stopping_criteria.update(self.epochs_summaries)
        self.latest_net_params[0] = self.latest_net_params[1]
        self.latest_net_params[1] = state_dict

    def view(self):
        pass  # TODO: ...

    def stop(self):
        return self.stopping_criteria.stop()

    def fix_weights(self, net):
        if isinstance(self.stopping_criteria, EarlyStopping):
            net.load_state_dict(self.latest_net_params[0])

    def print_last_epoch_summary(self):
        summary = self.epochs_summaries[-1]
        msg = "Epoch {}. ".format(len(self.epochs_summaries))
        if summary["val_acc"] is not None:
            msg += "Validation accuracy: {val_acc:2f}.".format(val_acc=summary["val_acc"])
        msg += "Train accuracy: {train_acc:.2f},  Train loss:  {train_loss:.2f}\n".format(train_acc=summary["acc"],
                                                                                          train_loss=summary["loss"])
        logger.log_print(msg)

    def restart(self):
        self.epoch_number = 0
        self.epochs_summaries = list()
        self.stopping_criteria.restart()
        self.latest_net_params = [None, None]


def train_nn(net, optimizer, loss_fn, train_dl, epochs: Epochs, attack=None, device=None, val_dl=None):
    """
    :param net: the network to train :)
    :param optimizer: torch.optim optimizer (e.g. SGD or Adam).
    :param loss_fn: we train with respect to this loss.
    :param train_dl: train data loader. We use that to iterate over the train data.
    :param val_dl: validation data loader. We use that to iterate over the validation data to measure performance.
                   None for ignore. If epochs uses Early Stopping then val_dl cannot be None.
    :param epochs: number of epochs to train or early stopping.
    :param attack: the attack we want to defense from - only PGD and FGSM are implemented. None for natural training
                   (i.e. with no resistance to any specific attack).
    :param device: use cuda or cpu
    """
    while not epochs.stop():
        batch_information_mat = []
        for batch_num, (batch_data, batch_labels) in enumerate(train_dl):
            if device is not None:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # train on natural batch
            if attack is None:
                batch_preds = net(batch_data)
                _loss = loss_fn(batch_preds, batch_labels)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

            # train on constructed adversarial examples (Adversarial Training Mode)
            else:
                with torch.no_grad():
                    batch_preds = net(batch_data)  # evaluate on natural examples

                adversarial_batch_data = attack.perturb(batch_data, batch_labels, device=device)
                adversarial_batch_preds = net(adversarial_batch_data)
                _loss = loss_fn(adversarial_batch_preds, batch_labels)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

            # calculate batch measurements
            hard_batch_preds = torch.argmax(batch_preds, dim=1)
            batch_num_currect = (hard_batch_preds == batch_labels).sum().item()
            batch_acc = batch_num_currect / len(batch_labels)
            batch_information_mat.append([_loss.item(), batch_acc])

        # summarize epoch (should be a part of the log):
        batch_information_mat = torch.tensor(batch_information_mat)
        emp_loss, emp_acc = torch.sum(batch_information_mat.T, dim=1) / len(batch_information_mat)
        if val_dl is not None:
            val_acc = measure_classification_accuracy(net, val_dl, device=device)
        else:
            val_acc = None
        curr_epoch_summary = {"acc": emp_acc, "loss": emp_loss, "val_acc": val_acc}
        epochs.update(curr_epoch_summary, net.state_dict())
        epochs.print_last_epoch_summary()

    epochs.fix_weights(net)
    # epochs.view() # save/plot epochs improvement


def measure_classification_accuracy(trained_net, dataloader: DataLoader, device=None):
    """
    couldn't load the whole dataset into GPU memory so the function calculates it in batches.
    :param dataloader: dataloader with data to measure on.
    :param trained_net: trained network to measure on dl.
    :return: accuracy (of trained_net on dl)
    """

    correct_classified = 0
    for xs, ys in dataloader:
        xs, ys = xs.to(device), ys.to(device)
        ys_pred = trained_net(xs)
        hard_ys_pred = torch.argmax(ys_pred, dim=1)
        correct_classified += (ys == hard_ys_pred).sum().item()
    return correct_classified / len(dataloader.dataset)
