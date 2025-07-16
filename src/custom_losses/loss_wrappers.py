import numpy as np
from typing import Tuple
from numdifftools import Derivative


class loss_wrapper:

    def __init__(
        self,
        loss_function,
        use_minus_loss_as_objective=True,
        clip=False,
        adaptive_weighting=False
    ):

        self.loss_function = loss_function
        self.sign = -1 if use_minus_loss_as_objective else 1
        self.clip = clip
        self.adaptive = adaptive_weighting

    def get_gradient(self, y_pred, y_true) -> Tuple[np.ndarray, np.ndarray]:
        y_true = y_true.get_label()

        def func(x):
            return self.loss_function(y_true, x)

        deriv1 = Derivative(func, n=1)(y_pred) * self.sign
        deriv2 = Derivative(func, n=2)(y_pred) * self.sign

        if self.clip:
            deriv1 = np.clip(deriv1, -1, 1)
            deriv2 = np.clip(deriv2, -1, 1)

        return deriv1, deriv2

    def get_metric(self, y_pred, y_true):
        y_true = y_true.get_label()
        if self.adaptive is False:
            loss = -np.mean(self.loss_function(y_true, y_pred))
        else:
            loss = -np.mean(self.loss_function(y_true, y_pred, training=False))

        return "loss", loss


class cat_wrapper:

    def __init__(
        self,
        loss_function,
        use_minus_loss_as_objective=True,
        clip=False,
        adaptive_weighting=False
    ):

        self.loss_function = loss_function
        self.sign = -1 if use_minus_loss_as_objective else 1
        self.clip = clip
        self.adaptive = adaptive_weighting

    def is_max_optimal(self) -> bool:
        return False

    def calc_ders_range(self, y_pred, y_true, weights=None) -> list[tuple[float, float]]:

        def func(x):
            return self.loss_function(y_true, x)

        deriv1 = Derivative(func, n=1, step=1e-6)(y_pred) * self.sign
        deriv2 = Derivative(func, n=2, step=1e-6)(y_pred) * self.sign

        if self.clip:
            deriv1 = np.clip(deriv1, -1, 1)
            deriv2 = np.clip(deriv2, -1, 1)

        result = [(d1, d2) for d1, d2 in zip(deriv1, deriv2)]

        return result
