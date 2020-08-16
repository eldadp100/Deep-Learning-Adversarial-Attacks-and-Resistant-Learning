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

    def test_attack(self, dataloader, plot_successful_attacks=True, main_title="", device=None):
        """
        the attack score of attack method A on network <net> is E[A(x) != y] over distribution D when A(x) is the
        constructed adversarial example of attack A on x. We are going to estimate it using samples from test_dataset.

        :param plot_successful_attacks: plot original vs adv imgs with additional information
        :return: the attack score.
        """

        # 1) calculate attack score
        self.net = self.net.to(device)
        num_successful_attacks = 0
        successful_attacks_details = []
        for i, (xs, ys) in enumerate(dataloader):
            if device is not None:
                xs, ys = xs.to(device), ys.to(device)

            if i > 1:
                break

            constructed_examples = self.perturb(xs, ys, device=device)
            y_preds = self.net(xs)
            hard_y_preds = torch.argmax(y_preds, dim=1)
            adv_y_preds = self.net(constructed_examples)
            hard_adv_y_preds = torch.argmax(adv_y_preds, dim=1)
            batch_successful_attack = (hard_adv_y_preds != hard_y_preds)
            num_successful_attacks += batch_successful_attack.sum().item()

            # update successful attacks details to plot
            if plot_successful_attacks:
                original_probs = torch.softmax(y_preds, dim=1)
                adv_probs = torch.softmax(adv_y_preds, dim=1)
                for i in range(len(batch_successful_attack)):
                    if len(successful_attacks_details) < configs.imgs_to_show and \
                            batch_successful_attack[i] and ys[i] == hard_y_preds[i]:
                        true_label = ys[i]
                        original_img = xs[i]
                        original_correct_prob = original_probs[i][true_label]
                        original_label = hard_y_preds[i]

                        adv_img = constructed_examples[i]
                        adv_correct_prob = adv_probs[i][true_label]
                        adv_label = hard_adv_y_preds[i]

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

        # 2) Visualization of the attack results
        if plot_successful_attacks:
            to_plot_imgs = []
            to_plot_titles = []
            to_plot_xlabels = []
            for img_description in successful_attacks_details:
                to_plot_imgs.extend(img_description[0])
                to_plot_titles.extend(img_description[1])
                to_plot_xlabels.extend(img_description[2])
            helper.show_img_lst(to_plot_imgs, to_plot_titles, to_plot_xlabels, main_title, columns=2)

        return attack_score, constructed_examples


class FGSM(Attack):
    def __init__(self, net, loss_fn, hp):
        """
        :param steps: number of steps in FGSM algorithm.
        :param epsilon: step size denoted as epsilon in the FGSM algorithm.
        """
        super().__init__(net, loss_fn)
        self.epsilon = hp["epsilon"]
        self.name = "fgsm"

    def perturb(self, X, y, device=None):
        X = X.to(device)
        y = y.to(device)
        X.requires_grad = True
        self.net = self.net.to(device)
        outputs = self.net(X)

        self.net.zero_grad()
        loss = self.loss_fn(outputs, y).to(device)
        loss.backward()

        attack_images = X + self.epsilon * X.grad.sign()
        attack_images = torch.clamp(attack_images, 0, 1)

        return attack_images








        # """
        # generates adversarial examples to given data points and labels (X, y) based on FGSM approach.
        # """
        # X_adv = torch.autograd.Variable(X, requires_grad=True).to(device)
        # y_pred = self.net(X_adv)
        # _loss = self.loss_fn(y_pred, y)
        # _loss.backward(retain_graph=True)
        # X_adv.retain_grad()
        # with torch.no_grad():
        #     X_adv += self.epsilon * torch.sign(X_adv.grad)
        # return X_adv


class PGD(Attack):
    def __init__(self, net, loss_fn, hp):
        """
        :param steps: number of steps in FGSM algorithm.
        :param epsilon: step size denoted as epsilon in the FGSM algorithm.
        """
        super().__init__(net, loss_fn)
        self.steps = hp["steps"]
        self.epsilon = hp["epsilon"]
        self.name = "pgd"

    def perturb(self, X, y, device=None):
        """
        generates adversarial examples to given data points and labels (X, y) based on PGD approach.
        """
        X_adv = torch.autograd.Variable(X, requires_grad=True).to(device)
        for j in range(self.steps):
            y_pred = self.net(X_adv)
            _loss = self.loss_fn(y_pred, y)
            _loss.backward(retain_graph=True)
            X_adv.retain_grad()
            with torch.no_grad():
                X_adv += self.epsilon * torch.sign(X_adv.grad)
                # the projection:
                Diff = torch.clamp(X_adv - X, -1, 1)  # l_inf projection on X (natural)
                X_adv = X + Diff
            if X_adv.grad is not None:
                X_adv.grad.zero_()
            X_adv.requires_grad_()

        return X_adv
