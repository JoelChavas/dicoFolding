# Third party import
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


class NTXenLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations,
    arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1),
                                    correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)
