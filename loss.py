import torch


class BinaryContrastiveLoss():
    def __init__(self, detach: bool = False):
        # TODO: implement a variable for the reduction operation (?)
        self._detach = detach

    def _get_by_indexing(self, features, indexes):
        indexes = indexes.expand(-1, -1, features.shape[-1])

        safe_indices = indexes.clamp(min=0)

        out = torch.gather(features, dim=1, index=safe_indices)

        if self._detach:
            out = out.detach()

        return out

    def __call__(self, features, positive_index, negative_index):
        device = features.device

        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        mask = (positive_index >= 0).squeeze(-1).to(device).float()

        positive_features = self._get_by_indexing(features, positive_index.to(device))
        negative_features = self._get_by_indexing(features, negative_index.to(device))

        pos_dist = torch.nn.functional.cosine_similarity(features, positive_features, dim=-1)
        neg_dist = torch.nn.functional.cosine_similarity(features, negative_features, dim=-1)
        distances = torch.stack((pos_dist, neg_dist), -1)

        distribution = distances.exp()

        loss = pos_dist.exp() / distribution.sum(-1)

        loss = -torch.log(loss)

        loss = loss * mask
        # remove the effected of padded tokens on the loss
        masked_loss = loss.sum(1) / mask.sum(-1)

        return masked_loss.mean()
