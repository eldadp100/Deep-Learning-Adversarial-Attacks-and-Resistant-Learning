import time
import torch
from torch.utils.data import DataLoader


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
    pass  # TODO: ...


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

    def update(self, epoch_summary):
        self.epoch_number += 1
        self.epochs_summaries.append(epoch_summary)
        self.stopping_criteria.update(self.epochs_summaries)

    def view(self):
        pass  # TODO: ...

    def stop(self):
        return self.stopping_criteria.stop()

    def print_last_epoch_summary(self):
        summary = self.epochs_summaries[-1]
        print("Epoch {i}\nAccuracy: {acc}\n loss: {loss}\n".format(i=len(self.epochs_summaries),
                                                                   acc=summary["acc"], loss=summary["loss"]))

    def restart(self):
        self.epoch_number = 0
        self.epochs_summaries = list()
        self.stopping_criteria.restart()


# TODO: should add log...
def train_nn(net, optimizer, loss_fn, dl, epochs: Epochs, attack=None, device=None):
    """
    :param net: the network to train :)
    :param optimizer: torch.optim optimizer (e.g. SGD or Adam).
    :param loss_fn: we train with respect to this loss.
    :param dl: data loader. We use that to iterate over the data.
    :param epochs: number of epochs to train or early stopping.
    :param attack: the attack we want to defense from - only PGD and FGSM are implemented. None for natural training
                   (i.e. with no resistance to any specific attack).
    :param device: use cuda or cpu
    """
    while not epochs.stop():
        batch_information_mat = []
        for batch_num, (batch_data, batch_labels) in enumerate(dl):
            if device is not None:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # generate adversarial examples:
            adversarial_batch_data = None
            if attack is not None:
                adversarial_batch_data = attack.perturb(batch_data, batch_labels,
                                                        device=device)  # Danskin's Theorem applies here.

            # train on natural batch
            if attack is None:  # TODO: maybe need to be done anyway? Theoretically not...
                batch_preds = net(batch_data)
                _loss = loss_fn(batch_preds, batch_labels)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

            # train on the constructed adversarial examples
            else:  # attack is not None
                batch_preds = net(adversarial_batch_data)
                _loss = loss_fn(batch_preds, batch_labels)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

            # calculate batch measurements
            hard_batch_preds = torch.argmax(batch_preds, dim=1)
            batch_num_currect = (hard_batch_preds == batch_labels).sum().type(torch.FloatTensor)
            batch_acc = torch.div(batch_num_currect, len(batch_labels))
            batch_information_mat.append([_loss.item(), batch_acc.item()])

        # summarize epoch (should be a part of the log):
        batch_information_mat = torch.tensor(batch_information_mat)
        emp_loss, emp_acc = torch.sum(batch_information_mat.T, dim=1) / len(batch_information_mat)
        curr_epoch_summary = {"acc": emp_acc, "loss": emp_loss}
        epochs.update(curr_epoch_summary)
        epochs.print_last_epoch_summary()


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
