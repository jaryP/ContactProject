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
        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        mask = (positive_index >= 0).squeeze(-1)
        non_masked_values = mask.sum(-1)

        positive_features = self._get_by_indexing(features, positive_index)
        negative_features = self._get_by_indexing(features, negative_index)

        # pos_dist = torch.norm(features - positive_features, p=2, dim=-1)
        # neg_dist = torch.norm(features - negative_features, p=2, dim=-1)

        pos_dist = torch.nn.functional.cosine_similarity(features, positive_features, dim=-1)
        neg_dist = torch.nn.functional.cosine_similarity(features, negative_features, dim=-1)
        # neg_dist = torch.norm(features - negative_features, p=2, dim=-1)
        distances = torch.stack((pos_dist, neg_dist), -1)

        # label = torch.as_tensor([1])
        # label[:, 0] = 1

        # loss = torch.nn.functional.cross_entropy(distances, label)

        distribution = distances.exp()

        loss = pos_dist.exp() / distribution.sum(-1)
        loss = -torch.log(loss)

        # pos_loss = torch.norm(features - positive_features, p=2, dim=-1)
        # neg_loss = 1 / (torch.norm(features - negative_features, p=2, dim=-1) + 1e-6)
        #
        # # masking out padded sequences
        # pos_loss = (pos_loss * mask).sum(-1) / non_masked_values
        #
        # neg_loss = (neg_loss * mask).sum(-1) / non_masked_values
        #
        # loss = -(pos_loss + neg_loss).log()

        return loss.mean()
