import torch.nn as nn

from typing import Optional

from torch import Tensor

from .transformer import (
    MultiHeadedAttention,
    PositionWiseFeedForward,
)


__all__ = (
    'CrossAttention',
    'CrossAttention2',
)


class CrossAttention(nn.Module):

    def __init__(self,
                 dim_model: int = 256,
                 dim_ff: int = 1024,
                 num_heads: int = 4,
                 dropout_prob: float = 0.1,
                 ):
        super().__init__()

        # params
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        # layers
        self.dropout = nn.Dropout(p=dropout_prob)
        self.attention = MultiHeadedAttention(
            num_heads=num_heads,
            dim_model=dim_model,
            dropout_prob=dropout_prob,
        )
        self.pwff = PositionWiseFeedForward(
            dim_model=dim_model,
            dim_ff=dim_ff,
            dropout_prob=dropout_prob,
        )

    def forward(self,
                E: Tensor,
                P: Tensor,
                mask: Optional[Tensor] = None,
                ):
        r = self.attention(E, P, P, mask=mask)
        r = self.dropout(r)
        Y = r * E
        Y = self.pwff(Y)
        return Y


class CrossAttention2(nn.Module):

    def __init__(self,
                 dim_model: int = 256,
                 dim_ff: int = 1024,
                 num_heads: int = 4,
                 dropout_prob: float = 0.1,
                 ):
        super().__init__()

        # params
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        # layers
        self.dropout = nn.Dropout(p=dropout_prob)
        self.attention = MultiHeadedAttention(
            num_heads=num_heads,
            dim_model=dim_model,
            dropout_prob=dropout_prob,
        )
        self.pwff = PositionWiseFeedForward(
            dim_model=dim_model,
            dim_ff=dim_ff,
            dropout_prob=dropout_prob,
        )

    def forward(self,
                residual: Tensor,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[Tensor] = None,
                ):
        r = self.attention(query, key, value, mask=mask)
        r = self.dropout(r)
        Y = r * residual
        Y = self.pwff(Y)
        return Y
