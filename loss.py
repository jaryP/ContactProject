import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    """
    A contrastive supervised loss, which attracts residues with contacts closer and move the other away
    """

    def __init__(self, average=True, detach: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detach = detach
        self.average = average

    def _get_by_indexing(self, features, indexes):
        indexes = indexes.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1])

        # remove padding indices (-1)
        safe_indices = indexes.clamp(min=0)

        out = torch.gather(features.unsqueeze(1).expand(-1, features.shape[1], -1, -1), 2, safe_indices)

        if self._detach:
            out = out.detach()

        return out

    def forward(self, features: torch.Tensor,
                pos_indexes: torch.Tensor, neg_indexes: torch.Tensor,
                temperature: float = 1.0, *args, **kwargs):

        device = features.device

        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        pos_indexes = pos_indexes.to(device)
        pos_mask = pos_indexes >= 0
        mask = pos_mask.any(-1).to(device).float()

        positive_features = self._get_by_indexing(features, pos_indexes)
        negative_features = self._get_by_indexing(features, neg_indexes.to(device))

        pos_dist = torch.nn.functional.cosine_similarity(features.unsqueeze(2), positive_features, dim=-1)
        neg_dist = torch.nn.functional.cosine_similarity(features.unsqueeze(2), negative_features, dim=-1)

        distances = torch.cat((pos_dist, neg_dist), -1) / temperature

        loss = pos_dist.exp() / distances.exp().sum(-1, keepdim=True)

        loss = -torch.log(loss).mean(-1)

        loss = loss * mask
        # remove the affected of padded tokens on the loss
        masked_loss = loss.sum(1) / mask.sum(-1)

        if self.average:
            masked_loss = masked_loss.mean()
        return masked_loss


class MaskedBCE(nn.Module):
    """
    A masked version of the BCEWithLogitsLoss, which also works protein wise by averaging the losses over them
    """

    def __init__(self, average=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inner_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.average = average

    def forward(self, logits: torch.Tensor, gt_matrix: torch.Tensor, *args, **kwargs):
        B = len(logits)

        gt_matrix = gt_matrix.to(logits.device)

        loss = self._inner_loss(logits.view(B, -1),
                                gt_matrix.view(B, -1))

        mask = (gt_matrix >= 0).float().view(B, -1)

        loss = loss * mask
        masked_loss = loss.sum(1) / mask.sum(-1)

        if self.average:
            masked_loss = masked_loss.mean()
        return masked_loss
