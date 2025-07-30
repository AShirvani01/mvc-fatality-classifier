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

    def __init__(self, y_true: np.ndarray, max_m=0.5, class_weight='balanced', s=30):

        super().__init__(y_true, class_weight)

        minority = np.sum(y_true)
        majority = len(y_true) - minority
        cls_num_list = torch.tensor([majority, minority])
        m_list = torch.pow(cls_num_list, -1/4)
        m_list *= max_m / torch.max(m_list)
        self.m_list = m_list

        assert s > 0
        self.s = s

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):

        index = y_true > 0
        batch_m = torch.where(index, self.m_list[1], self.m_list[0])
        y_m = y_pred - batch_m

        output = torch.where(index, y_m, y_pred)

        p = torch.sigmoid(self.s*output)
        pos_loss = torch.log(p) * self.alpha_m
        neg_loss = torch.log(1-p) * self.alpha_M

        return y_true * pos_loss + (1 - y_true) * neg_loss


###############################################################################


class Focal_loss(loss_function):
    """based on https://arxiv.org/pdf/1708.02002"""

    def __init__(self, y_true, gamma=2.0, class_weight='balanced'):

        super().__init__(y_true, class_weight)
        self.gamma = gamma

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):

        p = torch.sigmoid(y_pred)

        pos_loss = torch.pow(1-p, self.gamma) * torch.log(p) * self.alpha_m
        neg_loss = torch.pow(p, self.gamma) * torch.log(1-p) * self.alpha_M

        return y_true * pos_loss + (1 - y_true) * neg_loss
