import functools
from typing import Tuple

import torch
from esm.rotary_embedding import apply_rotary_pos_emb, RotaryEmbedding
from torch import nn


class OffsetRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, rope: RotaryEmbedding, *_, **__):
        super().__init__()

        self._rope = rope
        self.offsets = None

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self._rope, item)

    @functools.lru_cache(maxsize=500, typed=False)
    def get_embedding(self, position_offsets, shape, device='cpu'):

        batch_size, seq_len, _ = shape

        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        if position_offsets is not None:
            t = t.unsqueeze(0) + position_offsets.unsqueeze(1)

            freqs = torch.einsum("bi,j->bij", t, self.inv_freq)
        else:
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.offsets is None:
            return self._rope(q, k)
        else:
            shape = k.shape
            cos, sin = self.get_embedding(position_offsets=self.offsets, shape=shape, device=k.device)

            heads = shape[0] // len(self.offsets)
            cos = cos[None].expand(heads, -1, -1, -1).reshape(shape[0], *cos.shape[1:])
            sin = sin[None].expand(heads, -1, -1, -1).reshape(shape[0], *sin.shape[1:])

            self.offsets = None

            return (
                apply_rotary_pos_emb(q, cos, sin),
                apply_rotary_pos_emb(k, cos, sin),
            )


class ContactsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model

    def forward(self, input_ids, **kwargs):
        return self.backbone.predict_contacts(input_ids)


class ContrastiveWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
        self.repr_layers = model.num_layers

    def forward(self, input_ids, **kwargs):
        h = self.backbone(input_ids, repr_layers=[self.repr_layers])
        h = h['representations'][self.repr_layers][:, 1:-1]

        if self.training:
            return h
        else:
            logits = torch.nn.functional.cosine_similarity(h[..., None, :, :], h[..., :, None, :], dim=-1)

            # rescale in range [0, 1] for compatibility purposes
            logits = (logits + 1)/2
            return logits
