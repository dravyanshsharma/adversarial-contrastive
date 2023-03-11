import numpy as np
import torch
from torch import nn
import math
import torch.nn.functional as F


class IntraBatchContrast(nn.Module):
    def __init__(self, temperature=0.1, batch_size=512):
        super(IntraBatchContrast, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.mask_pos_repr, self.mask_neg_repr = self._get_correlated_mask()

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l_up = np.eye(2 * self.batch_size, 2 * self.batch_size, k = self.batch_size)
        l_low = np.eye(2 * self.batch_size, 2 * self.batch_size, k = -self.batch_size)
        mask_pos = torch.from_numpy((l_up + l_low))
        mask_pos = mask_pos.type(torch.bool)
        mask_neg = torch.from_numpy((diag + l_up + l_low))
        mask_neg = (1 - mask_neg).type(torch.bool)
        return mask_pos.cuda(non_blocking=True), mask_neg.cuda(non_blocking=True)
    
    def forward(self, representations):
        # calculate similarity
        distance_matrix = 2.0 - 2.0 * torch.mm(representations, representations.t())
        similarity_matrix = -distance_matrix

        # filter out the scores from the positive samples: (4N, 3)
        positives = similarity_matrix[self.mask_pos_repr].view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_neg_repr].view(2 * self.batch_size, -1)


        return torch.cat((positives, negatives), dim=1).div(self.temperature)





