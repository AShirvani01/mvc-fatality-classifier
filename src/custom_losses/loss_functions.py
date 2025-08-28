import numpy as np
import torch


class loss_function:

    def __init__(self, y_true: np.ndarray, class_weight='balanced'):

        if class_weight == 'balanced':
            minority = np.sum(y_true)
            majority = len(y_true) - minority

            majority = (1 / majority) * (len(y_true)/2)
            minority = (1 / minority) * (len(y_true)/2)
            self.alpha_m = minority / (majority + minority)
            self.alpha_M = majority / (majority + minority)
        else:
            self.alpha_m = 1
            self.alpha_M = 1


###############################################################################


class LDAM_loss(loss_function):
    """based on https://arxiv.org/pdf/1906.07413"""

    def __init__(self, y_true: np.ndarray, c=1.0, class_weight='balanced'):

        super().__init__(y_true, class_weight)

        assert c > 0
        n_1 = np.sum(y_true)
        n_0 = len(y_true) - n_1
        n_list = torch.tensor([n_0, n_1])
        n_list = c * torch.pow(n_list, -1/4)
        self.n_list = n_list

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):

        p_1 = torch.sigmoid(y_pred - self.n_list[1])
        p_0 = torch.sigmoid(y_pred + self.n_list[0])

        pos_loss = torch.log(p_1) * self.alpha_m
        neg_loss = torch.log(1-p_0) * self.alpha_M

        return y_true * pos_loss + (1 - y_true) * neg_loss


###############################################################################


class Focal_loss(loss_function):
    """based on https://arxiv.org/pdf/1708.02002"""

    def __init__(self, y_true: np.ndarray, gamma=2.0, class_weight='balanced'):

        super().__init__(y_true, class_weight)
        self.gamma = gamma

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):

        p = torch.sigmoid(y_pred)

        pos_loss = torch.pow(1-p, self.gamma) * torch.log(p) * self.alpha_m
        neg_loss = torch.pow(p, self.gamma) * torch.log(1-p) * self.alpha_M

        return y_true * pos_loss + (1 - y_true) * neg_loss


###############################################################################


class LA_loss(loss_function):
    "based on https://arxiv.org/pdf/2007.07314"

    def __init__(self, y_true: np.ndarray, tau=1.0, class_weight='balanced'):

        super().__init__(y_true, class_weight)

        n = len(y_true)
        n_1 = np.sum(y_true)
        n_0 = n - n_1
        pi_1 = n_1 / n
        pi_0 = n_0 / n

        self.offset = tau * np.log(pi_1 / pi_0)

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):

        p = torch.sigmoid(y_pred + self.offset)

        pos_loss = torch.log(p) * self.alpha_m
        neg_loss = torch.log(1-p) * self.alpha_M

        return y_true * pos_loss + (1 - y_true) * neg_loss
