import torch


class BinaryContrastiveLoss():
    def __init__(self, detach: bool = False):
        # TODO: implement a variable for the reduction operation (?)
        self._detach = detach

    def _get_by_indexing(self, features, indexes):
        # indexes = indexes.expand(-1, -1, features.shape[-1])

        indexes = indexes.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1])

        # remove padding indices (-1)
        safe_indices = indexes.clamp(min=0)

        # out = torch.gather(features, dim=1, index=safe_indices)

        out = torch.gather(features.unsqueeze(1).expand(-1, features.shape[1], -1, -1), 2, safe_indices)

        if self._detach:
            out = out.detach()

        return out

    def __call__(self, features, positive_index, negative_index):
        device = features.device

        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        mask = (positive_index >= 0).any(-1).to(device).float()

        positive_features = self._get_by_indexing(features, positive_index.to(device))
        negative_features = self._get_by_indexing(features, negative_index.to(device))

        pos_dist = torch.nn.functional.cosine_similarity(features.unsqueeze(2), positive_features, dim=-1)
        neg_dist = torch.nn.functional.cosine_similarity(features.unsqueeze(2), negative_features, dim=-1)
        distances = torch.cat((pos_dist, neg_dist), -1)

        distribution = distances.exp()

        loss = pos_dist.exp() / distribution.sum(-1, keepdim=True)

        loss = -torch.log(loss)
        # sum over the positive residuals
        loss = loss.mean(-1)

        loss = loss * mask
        # remove the effected of padded tokens on the loss
        masked_loss = loss.sum(1) / mask.sum(-1)

        return masked_loss.mean()
