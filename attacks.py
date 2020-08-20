import os

import torch
import configs
import helper


class Attack:
    def __init__(self, net, loss_fn):
        """
        :param net: the network to attack
        :param loss_fn: the attack is with respect to this loss
        """
        self.net = net
        self.loss_fn = loss_fn
        self.name = None

    def perturb(self, X, y, device=None):
        pass

    def test_attack(self, dataloader, plot_results=True, save_results_figs=True, fig_path=None, main_title="",
                    device=None):
        """
        the attack score of attack method A on network <net> is E[A(x) != y] over distribution D when A(x) is the
        constructed adversarial example of attack A on x. We are going to estimate it using samples from test_dataset.

        :param plot_results: plot original vs adv imgs with additional information
        :return: the attack score.
        """
        # calculate attack score
        self.net = self.net.to(device)  # move network to device
        num_successful_attacks = 0
        successful_attacks_details = []
        for batch_num, (xs, ys) in enumerate(dataloader):
            if device is not None:
                xs, ys = xs.to(device), ys.to(device)  # move data to device

            # calculate %successful attacks on xs, ys
            constructed_examples = self.perturb(xs, ys, device=device)
            y_preds = self.net(xs)
            hard_y_preds = torch.argmax(y_preds, dim=1)
            adv_y_preds = self.net(constructed_examples)
            hard_adv_y_preds = torch.argmax(adv_y_preds, dim=1)
            batch_successful_attack = (hard_adv_y_preds != hard_y_preds)
            num_successful_attacks += batch_successful_attack.sum().item()

            # update successful attacks details to plot
            if plot_results or save_results_figs:
                original_probs = torch.softmax(y_preds, dim=1)
                adv_probs = torch.softmax(adv_y_preds, dim=1)
                for idx in range(len(batch_successful_attack)):
                    if len(successful_attacks_details) < configs.imgs_to_show and \
                            batch_successful_attack[idx] and ys[idx] == hard_y_preds[idx]:
                        true_label = ys[idx]
                        original_img = xs[idx]
                        original_correct_prob = original_probs[idx][true_label]
                        original_label = hard_y_preds[idx]

                        adv_img = constructed_examples[idx]
                        adv_correct_prob = adv_probs[idx][true_label]
                        adv_label = hard_adv_y_preds[idx]

                        x_label_txt_to_format = "correct_prob:{p}\npred label:{l}\ntrue label:{r}"
                        successful_attacks_details.append((
                            [original_img, adv_img],  # imgs
                            ["Original".format(), "Constructed Adversarial Example"],  # titles
                            # x_labels (txt below imgs):
                            [
                                x_label_txt_to_format.format(p=original_correct_prob, l=original_label, r=true_label),
                                x_label_txt_to_format.format(p=adv_correct_prob, l=adv_label, r=true_label)
                            ],
                            main_title  # main title
                        ))
        attack_score = num_successful_attacks / len(dataloader.dataset)

        # visualize attack results
        if plot_results or save_results_figs:
            to_plot_imgs = []
            to_plot_titles = []
            to_plot_xlabels = []
            for img_description in successful_attacks_details:
                to_plot_imgs.extend(img_description[0])
                to_plot_titles.extend(img_description[1])
                to_plot_xlabels.extend(img_description[2])

            if len(to_plot_imgs) > 0:
                helper.show_img_lst(to_plot_imgs, to_plot_titles, to_plot_xlabels, main_title, columns=2,
                                    plot_img=plot_results, save_img=save_results_figs, save_path=fig_path)

        return attack_score


class FGSM(Attack):
    def __init__(self, net, loss_fn, hp):
        """
        :param net:
        :param loss_fn:
        :param hp:
        """
        super().__init__(net, loss_fn)
        self.epsilon = hp["epsilon"]
        self.name = "fgsm"

    def perturb(self, X, y, device=None):
        """
        generates adversarial examples to given data points and labels (X, y) based on FGSM approach.
        :param X:
        :param y:
        :param device:
        :return:
        """

        X.requires_grad = True
        y_pred = self.net(X)

        self.net.zero_grad()
        loss = self.loss_fn(y_pred, y).to(device)
        loss.backward()

        adv_X = X + self.epsilon * X.grad.sign()
        adv_X = torch.clamp(adv_X, 0, 1)

        return adv_X


class PGD(Attack):
    def __init__(self, net, loss_fn, hp):
        """
        :param net:
        :param loss_fn:
        :param hp:
        """
        super().__init__(net, loss_fn)
        self.steps = hp["steps"]
        self.alpha = hp["alpha"]
        self.epsilon = hp["epsilon"]
        self.name = "pgd"

    def perturb(self, X, y, device=None):
        """
        generates adversarial examples to given data points and labels (X, y) based on PGD approach.
        :param X:
        :param y:
        :param device:
        :return:
        """

        original_X = X
        for i in range(self.steps):
            X.requires_grad_()
            outputs = self.net(X)
            _loss = self.loss_fn(outputs, y).to(device)
            _loss.backward()

            X = X + self.alpha * X.grad.sign()
            diff = torch.clamp(X - original_X, min=-self.epsilon, max=self.epsilon)  # gradient projection
            X = torch.clamp(original_X + diff, min=0.0, max=1.0).detach_()  # to stay in image range [0,1]

        return X


class MomentumFGSM(Attack):
    """ Momentum Iterative Fast Gradient Sign Method (https://arxiv.org/pdf/1710.06081.pdf) """

    def __init__(self, net, loss_fn, hp):
        """
        :param net:
        :param loss_fn:
        :param hp:
        """
        super().__init__(net, loss_fn)
        self.steps = hp["steps"]
        self.alpha = hp["alpha"]
        self.momentum = hp["momentum"]
        self.epsilon = hp["epsilon"]
        self.name = "pgd"

    def perturb(self, X, y, device=None):
        """
        generates adversarial examples to given data points and labels (X, y) based on PGD approach.
        :param X:
        :param y:
        :param device:
        :return:
        """

        original_X = X
        accumulated_grad = None
        for i in range(self.steps):
            X.requires_grad_()
            outputs = self.net(X)
            _loss = self.loss_fn(outputs, y).to(device)
            _loss.backward()

            accumulated_grad = X.grad if accumulated_grad is None else self.momentum * accumulated_grad + (
                    1.0 - self.momentum) * X.grad
            X = X + self.alpha * accumulated_grad.sign()
            diff = torch.clamp(X - original_X, min=-self.epsilon, max=self.epsilon)  # gradient projection
            X = torch.clamp(original_X + diff, min=0.0, max=1.0).detach_()  # to stay in image range [0,1]

        return X
