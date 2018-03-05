import numpy
import torch
import torch.nn as nn
import torch.nn.functional as func

class LossFunc(nn.Module):
    def __init__(self, f1=False, f2=False):
        super(LossFunc, self).__init__()
        self.f1 = f1
        self.f2 = f2

    def forward(self, x, y, W, alpha, rho, gamma=None, tau=None):
        norm2sqr = lambda a: torch.pow(a.norm(p=2), 2)
        dis2sqr = lambda a, b: torch.pow(func.pairwise_distance(a, b), 2)
        # print('W: ', W.size(), 'tau: ', tau.size())

        term_1 = dis2sqr(x.mm(W), y) / alpha
        if self.f1:
            p = (gamma * W).norm(p=1)
            term_2 = (alpha * rho * ((gamma * W).norm(p=1))
                      + alpha * (1 - rho) * 0.5 * norm2sqr(gamma * W))
        else:
            term_2 = (alpha * rho * W.norm(p=1)
                      + alpha * (1 - rho) * 0.5 * norm2sqr(W))

        if self.f2:
#            term_3 = 0
            n = W.size(0)
            term_3 = tau.t().mm(torch.pow(W[:-1].abs() - W[1:].abs(), 2))
#            for j in range(n - 1):
#                term_3 = term_3 + tau[j] * ((W[j].abs() - W[j + 1].abs())
#                                             * (W[j].abs() - W[j + 1].abs()))
            loss = term_1 + term_2 + term_3
        else:
            loss = term_1 + term_2

        loss = loss.mean()

        return loss
