import numpy as np
from typing import Tuple
import torch
from torch.autograd import grad


class loss_wrapper:

    def __init__(
        self,
        loss_function,
        clip=False
    ):

        self.loss_function = loss_function
        self.clip = clip

    def get_gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        y_pred = torch.tensor(y_pred, requires_grad=True)

        if hasattr(y_true, 'get_label'):
            y_true = torch.tensor(y_true.get_label())
        else:
            y_true = torch.tensor(y_true)

        loss = -self.loss_function(y_true, y_pred).sum()

        gradient = grad(loss, y_pred, create_graph=True)[0]

        hessian = torch.zeros_like(y_pred)
        for i in range(y_pred.numel()):
            hessian[i] = grad(gradient[i], y_pred, create_graph=True)[0][i]

        if self.clip:
            gradient = torch.clamp(gradient, -1, 1)
            hessian = torch.clamp(hessian, -1, 1)

        return gradient.detach().numpy(), hessian.detach().numpy()


class cat_wrapper:

    def __init__(
        self,
        loss_function,
        clip=False
    ):

        self.loss_function = loss_function
        self.clip = clip

    def calc_ders_range(self, y_pred: np.ndarray, y_true: np.ndarray, weights=None) -> list[tuple[float, float]]:
        """
        - Assuming each observation is independent, only take diag(hessian)
        - CatBoost doesn't apply negative to result internally like
        XGBoost/LightGBM -> need to apply manually
        """

        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred, requires_grad=True)

        loss = -self.loss_function(y_true, y_pred).sum()

        gradient = grad(loss, y_pred, create_graph=True)[0]  # Shape(y_pred, )

        hessian = torch.zeros_like(y_pred)
        for i in range(y_pred.numel()):
            hessian[i] = grad(gradient[i], y_pred, create_graph=True)[0][i]  # Shape(y_pred, )

        if self.clip:
            gradient = torch.clamp(gradient, -1, 1)
            hessian = torch.clamp(hessian, -1, 1)

        gradient = gradient.detach().numpy()
        hessian = hessian.detach().numpy()

        return [(grad, hess) for grad, hess in zip(-gradient, -hessian)]
