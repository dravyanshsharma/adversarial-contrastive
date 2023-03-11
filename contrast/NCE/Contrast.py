import numpy as np
import torch
from torch import nn
import math


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, feature_dim, queue_size, temperature=0.07):
        super(MemoryMoCo, self).__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.index = 0

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)

    def forward(self, q, k, k_all):
        k = k.detach()

        l_pos = (q * k).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        # TODO: remove clone. need update memory in backwards
        l_neg = torch.mm(q, self.memory.clone().detach().t())
        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, self.temperature).contiguous()

        # update memory
        with torch.no_grad():
            all_size = k_all.shape[0]
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
            self.memory.index_copy_(0, out_ids, k_all)
            self.index = (self.index + all_size) % self.queue_size

        return out


class IntraBatchContrast(nn.Module):
    def __init__(self, temperature=0.07, batch_size=512):
        super(IntraBatchContrast, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.mask_same_repr = self._get_correlated_mask()

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.cuda(non_blocking=True)
    
    def forward(self, representations):
        similarity_matrix = torch.mm(representations, representations.t())

        # filter out the scores from the positive samples: (2N, 1)
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # negatives: (2N, 2N-2)
        negatives = similarity_matrix[self.mask_same_repr].view(2 * self.batch_size, -1)

        return torch.cat((positives, negatives), dim=1).div(self.temperature)