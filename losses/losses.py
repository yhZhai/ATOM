from typing import List

import torch
import torch.nn as nn


class Losses(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.l2_loss = nn.MSELoss(reduction="none")

    def _sum_flat(self, tensor):
        """
        Take the sum over all non-batch dimensions.
        """
        return tensor.sum(dim=list(range(1, len(tensor.shape))))

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = self._sum_flat(
            loss * mask.float()
        )  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = self._sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val

    def kl_loss(self, mu, logvar):
        loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0
        )
        return loss

    def att_spa_loss(self, att_list: List[torch.Tensor]):
        loss = 0
        for att in att_list:
            # loss = loss - att.max(dim=2)[0].sum(dim=1).mean()
            loss = loss - att.max(dim=2)[0].mean()  # TODO not sure
        loss = loss / len(att_list)
        return loss

    def codebook_norm_loss(self, codebook):
        sim_mat = torch.matmul(codebook, codebook.transpose(0, 1))
        identity = torch.eye(codebook.shape[0]).to(sim_mat.device)
        diff = sim_mat - identity
        norm = torch.linalg.norm(diff)
        return norm
