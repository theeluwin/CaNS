import numpy as np
import torch.nn as nn

from typing import Optional

from torch import (
    bmm as torch_bmm,
    Tensor,
    LongTensor,
)
from torch.nn.functional import normalize as F_normalize

from tools.utils import fix_random_seed

from .encoders import AdvancedItemEncoder


__all__ = (
    'GRU4RecPP',
)


class GRU4RecPP(nn.Module):

    def __init__(self,
                 num_items: int,
                 ifeatures: np.ndarray,
                 ifeature_dim: int,
                 icontext_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 dropout_prob: float = 0.1,
                 random_seed: Optional[int] = None
                 ):
        """
            Note that item index starts from 1.
            Use 0 label (ignore index in CE) to avoid learning unmasked(context, known) items.
        """
        super().__init__()

        # data params
        self.num_items = num_items
        self.ifeature_dim = ifeature_dim
        self.icontext_dim = icontext_dim

        # main params
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # optional params
        self.dropout_prob = dropout_prob
        self.random_seed = random_seed

        # set seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # main layers
        self.item_encoder = AdvancedItemEncoder(
            num_items=num_items,
            ifeatures=ifeatures,
            ifeature_dim=ifeature_dim,
            icontext_dim=icontext_dim,
            hidden_dim=hidden_dim,
            random_seed=random_seed,
        )
        self.grus = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )

    def forward(self,
                profile_tokens: LongTensor,  # (b x L)
                profile_icontexts: Tensor,  # (b x L x d_Ci)
                extract_tokens: LongTensor,  # (b x C)
                extract_icontexts: Tensor,  # (b x C x d_Ci)
                ):

        # get profile vectors
        # dim: (b x L x d)
        P = self.item_encoder(
            profile_tokens,
            profile_icontexts,
        )

        # get extract vectors
        # dim: (b x C x d)
        E = self.item_encoder(
            extract_tokens,
            extract_icontexts,
        )
        E = F_normalize(E, p=2, dim=2)

        # apply GRU
        # [process] (b x L x d) -> ... -> (b x L x d)
        P, _ = self.grus(P)
        P = F_normalize(P, p=2, dim=2)

        # get scores
        # dim: (b x C x d) @ (b x d x 1) -> (b x C)
        P = P[:, -1, :].unsqueeze(2)
        logits = torch_bmm(E, P).squeeze(2)

        return logits
