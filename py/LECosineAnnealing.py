#-*-coding:utf-8-*-
"""
    Linear Exponential Cosine Annealing Smooth Warm Restart Learning Rate for lr_scheduler.LambdaLR
    @author (Enigmatisms) HQY
    @date 2021.11.30
    @copyright Enigmatisms
"""

from math import cos
import math


class LECosineAnnealingSmoothRestart:
    """
        The Maximum lr is bounded by a linear function, while the mininum lr is bounded by a exponential function
        The frequency decreases over epochs, at (epochs: which is last_epoch) time, lr comes to the mininum
        - max_start (min_start): upper (lower) bound starting lr
        - max_end (min_end): upper (lower) bound ending lr
        - epochs: The number of steps lr_scheduler needs
        - folds: (upper bound of) number of peaks
        - use_linear: whether upper bound decreases linearly
    """
    def __init__(self, max_start, max_end, min_start, min_end, epochs, folds = 15, use_linear = False) -> None:
        coeff = (min_end / min_start) ** (1.0 / epochs)
        coeff2 = (max_end / max_start) ** (1.0 / epochs)
        b = epochs / (folds * 2.5 * math.pi)
        k = math.ceil(0.625 * folds - 0.25)
        a = 1 / (((k << 1) + 1) * math.pi) - 1 / (2.5 * math.pi * folds)
        self.f = None
        if use_linear: self.f = lambda x: (max_end - max_start) / epochs * x + max_start
        else: self.f = lambda x: max_start * (coeff2 ** x)
        self.g = lambda x: min_start * (coeff ** x)
        self.c = lambda x: cos(x / (a * x + b))

    def lr(self, x):
        return 0.5 * (self.f(x) - self.g(x)) * self.c(x) + (self.f(x) + self.g(x)) * 0.5       

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torch.optim import lr_scheduler

    opt = torch.optim.Adam(torch.nn.Conv2d(3, 3, 3, 1, 1).parameters(), lr = 1.0)
    def lr_sch_res(lr_func, epoch):
        sch = lr_scheduler.LambdaLR(opt, lr_func)
        result = []
        for _ in range(epoch):
            opt.step()
            sch.step()
            result.append(sch.get_last_lr()[-1])
        return np.array(result)

    max_ep = 30 * 782
    max_start = 5e-4
    max_end = 5e-5
    min_start = 1e-4
    min_end = 5e-7
    fold = 7

    xs = np.linspace(0, max_ep, max_ep + 1)
    inf_b = np.array([min_end for _ in xs])
    sup_b = np.array([max_start for _ in xs])
    lr_linear = LECosineAnnealingSmoothRestart(max_start, max_end, min_start, min_end, max_ep, fold, True)
    lr_exp = LECosineAnnealingSmoothRestart(max_start, max_end, min_start, min_end, max_ep, fold)
    ys1 = lr_sch_res(lr_linear.lr, max_ep + 1)
    ys2 = lr_sch_res(lr_exp.lr, max_ep + 1)
    plt.plot(xs, ys1, c = 'r', label = 'linear')
    plt.plot(xs, ys2, c = 'b', label = 'exp')
    plt.plot(xs, inf_b, c = 'grey', label = 'inf_b', linestyle='--')
    plt.plot(xs, sup_b, c = 'black', label = 'sup_b', linestyle='--')
    plt.grid(axis = 'both')
    plt.legend()
    plt.show()
