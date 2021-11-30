#-*-coding:utf-8-*-
"""
    Linear Exponential Cosine Annealing Smooth Warm Restart Learning Rate for lr_scheduler.LambdaLR
    @author (Enigmatisms) HQY
    @date 2021.11.30
    @copyright Enigmatisms
"""

from math import cos
import math

"""
    The Maximum lr is bounded by a linear function, while the mininum lr is bounded by a exponential function
    The frequency decreases over epochs, at (epochs: which is last_epoch) time, lr comes to the mininum
"""
class LECosineAnnealingSmoothRestart:
    def __init__(self, max_start, max_end, min_start, min_end, epochs, folds = 15) -> None:
        coeff = (min_end / min_start) ** (1.0 / epochs)
        b = epochs / (folds * 2.5 * math.pi)
        k = math.ceil(0.625 * folds - 0.25)
        a = 1 / (((k << 1) + 1) * math.pi) - 1 / (2.5 * math.pi * folds)
        self.f = lambda x: (max_end - max_start) / epochs * x + max_start
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

    max_ep = 20000
    max_start = 5e-4
    max_end = 5e-5
    min_start = 2e-4
    min_end = 5e-7
    fold = 8

    xs = np.linspace(0, max_ep, max_ep + 1)
    inf_b = np.array([min_end for _ in xs])
    sup_b = np.array([max_start for _ in xs])
    lr_class = LECosineAnnealingSmoothRestart(max_start, max_end, min_start, min_end, max_ep, fold)
    ys = lr_sch_res(lr_class.lr, max_ep + 1)
    plt.plot(xs, ys, c = 'k', label = 'lr')
    plt.plot(xs, inf_b, c = 'r', label = 'inf_b', linestyle='--')
    plt.plot(xs, sup_b, c = 'b', label = 'sup_b', linestyle='--')
    plt.grid(axis = 'both')
    plt.show()
