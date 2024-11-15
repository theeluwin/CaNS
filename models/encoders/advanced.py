import numpy as np
import torch.nn as nn

from typing import Optional

from torch import (
    cat as torch_cat,
    Tensor,
    LongTensor,
)

from tools.utils import fix_random_seed

from ..layers import TokenEmbedding


__all__ = (
    'AdvancedItemEncoder',
)


class AdvancedItemEncoder(nn.Module):

    def __init__(self,
                 num_items: int,
                 ifeatures: np.ndarray,
                 ifeature_dim: int,
                 icontext_dim: int,
                 hidden_dim: int = 64,
                 random_seed: Optional[int] = None,
                 special: Optional[str] = None,
                 ):
        super().__init__()

        # data params
        self.num_items = num_items
        self.ifeature_dim = ifeature_dim
        self.icontext_dim = icontext_dim

        # main params
        self.hidden_dim = hidden_dim

        # optional params
        self.random_seed = random_seed
        self.special = special

        # set seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # 0: padding token
        # 1 ~ V: item tokens
        self.vocab_size = num_items + 1

        # ifeature cache
        if ifeatures is not None:
            self.ifeature_cache = nn.Embedding.from_pretrained(Tensor(ifeatures), freeze=True)
        else:
            self.ifeature_cache = None

        # embedding layers
        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=hidden_dim
        )

        # main layers
        if ifeatures is not None:
            self.ac_encoder = nn.Linear(ifeature_dim + icontext_dim, hidden_dim * 4)
            self.item_encoder = nn.Linear(hidden_dim + hidden_dim * 4, hidden_dim)
        else:
            self.ac_encoder = nn.Linear(icontext_dim, hidden_dim)
            self.item_encoder = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

    def forward(self,
                tokens: LongTensor,  # (b x L|C)
                icontexts: Tensor,  # (b x L|C x d_Ci)
                ):

        if self.ifeature_cache is not None:

            # get ifeatures from cache
            ifeatures = self.ifeature_cache(tokens)

            # get ac vector
            ac = torch_cat([ifeatures, icontexts], dim=-1)
            ac_vector = self.ac_encoder(ac)

        else:

            # get ac vector
            ac_vector = self.ac_encoder(icontexts)

        # get token vector
        token_vector = self.token_embedding(tokens)

        # get item vector
        vector = torch_cat([token_vector, ac_vector], dim=-1)
        vector = self.item_encoder(vector)

        return vector
