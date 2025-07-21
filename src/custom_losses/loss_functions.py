import numpy as np
from scipy import optimize as opt
import torch
import torch.nn.functional as F


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
    "based on https://github.com/kaidic/LDAM-DRW"

    def __init__(self, y_true: np.ndarray, max_m=0.5, class_weight='balanced', s=30):

        super().__init__(y_true, class_weight)

        minority = np.sum(y_true)
        majority = len(y_true) - minority
        cls_num_list = [majority, minority]
        m_list = np.power(cls_num_list, -1/4)
        m_list *= max_m / np.max(m_list)
        self.m_list = m_list

        assert s > 0
        self.s = s

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        y_pred = torch.stack([y_pred, torch.where(y_pred == 0, 1, 0)], dim=1)  # [N,2]
        y_true = torch.stack([y_true, torch.where(y_true == 0, 1, 0)], dim=1)
        index = torch.zeros_like(y_pred, dtype=torch.uint8)
        index.scatter_(1, y_true, 1)
        index_float = index.type(torch.FloatTensor)

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        """
        index = y_true > 0
        batch_m = torch.where(index, self.m_list[1], self.m_list[0])
        y_m = y_pred - batch_m

        output = torch.where(index, y_m, y_pred)

        p = torch.sigmoid(self.s*output)
        pos_loss = torch.log(p) * self.alpha_m
        neg_loss = torch.log(1-p) * self.alpha_M

        return y_true * pos_loss + (1 - y_true) * neg_loss
