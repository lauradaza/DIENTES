import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss

from einops import rearrange
from src.utils.utils import get_key
from src.utils.constants import TEMPLATE


class ntBXentLoss(nn.Module):
    """Multi-label loss for contrastive learning. Adapted from:
    https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb
    """

    def __init__(
        self, temperature=0.1, batch_size=4096, prototypes=True, multiple_pos=True
    ) -> None:
        super().__init__()
        # SimCLR: with larger batches (2048, 4096), temperature towards 0.1
        self.temperature = temperature
        self.batch_size = batch_size

        self.similarity = self.sim_pixels if not prototypes else self.sim_prototypes
        self.loss = self.multiple_pos if multiple_pos else self.single_pos

    def get_pos_indices(self, y):
        # Select [self.batch_size] random positive positions
        pos_indices = torch.stack(torch.where(y))
        if len(pos_indices[0]) > self.batch_size:
            weights = (y.sum() - y.sum(1)) / y.sum(1)  # Neg / pos
            weights = torch.nan_to_num(weights)
            weights = (
                F.one_hot(pos_indices[0], num_classes=len(weights)) * weights
            ).sum(1)

            idxs = torch.multinomial(weights, self.batch_size)
            pos_indices = pos_indices[:, idxs]
        return pos_indices

    def add_bg_pos(self, y, mask):
        # if only 1 fg class, we need to use the bg as negatives
        y_neg = y.clone()
        y = y * mask

        y_neg = y - torch.roll(y_neg, shifts=(10, 10, 5), dims=(2, 3, 4))
        y_neg = (y_neg == -1).to(y.dtype)  # 1 fg, 0 intersection, -1 bg
        y = torch.cat([y_neg, y], 1)  # add bg positions close to fg
        return y

    def visualize_pos(self, y, pos_indices, B=1, C=1):
        selected = y.clone()
        selected[:, pos_indices[1]] = 5
        selected = rearrange(
            selected, "c (b h w d) -> b c h w d", b=4, h=96, w=96, d=96
        )
        import numpy as np
        import nibabel as nib

        nib.save(
            nib.Nifti1Image(selected[B, C].cpu().numpy(), np.eye(4)),
            f"test{B}_{C}.nii.gz",
        )

    def sim_pixels(self, x, y, mask):
        if y.shape[1] == 1:
            y = self.add_bg_pos(y, mask)

            y = rearrange(y, "b c h w d -> c (b h w d)")
            pos_indices = self.get_pos_indices(y)

            target = torch.eye(len(pos_indices[0])).to(pos_indices)
            target[pos_indices[0].bool()] = pos_indices[0]
            target = target.float()
        else:
            y = rearrange(y * mask, "b c h w d -> c (b h w d)")
            pos_indices = self.get_pos_indices(y)

            target = F.one_hot(pos_indices[0])
            target = torch.matmul(target, target.T).float()

        x = rearrange(x, "b c h w d -> c (b h w d)")
        x = x[:, pos_indices[1]].T  # [N, Ch]

        # Cosine similarity
        xcs = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)

        # Set logit of diagonal element to "inf" signifying complete
        # correlation. sigmoid(inf) = 1.0 so this will work out nicely
        # when computing the Binary Cross Entropy Loss.
        xcs[torch.eye(x.size(0)).bool()] = float("inf")
        return xcs, target

    def sim_prototypes(self, x, y, mask):
        if y.shape[1] == 1:
            y = self.add_bg_pos(y, mask)

            y = rearrange(y, "b c h w d -> c (b h w d)")
            pos_indices = self.get_pos_indices(y)
        else:
            y = rearrange(y * mask, "b c h w d -> c (b h w d)")
            pos_indices = self.get_pos_indices(y)

        x = rearrange(x, "b c h w d -> c (b h w d)")

        prototypes = torch.matmul(y, x.T).float()  # [Ch, Classes]
        prototypes /= y.sum(1)[:, None]
        target = F.one_hot(pos_indices[0]).float()

        x = x[:, pos_indices[1]].T  # [N, Ch]

        # Cosine similarity
        xcs = F.cosine_similarity(prototypes[None, :, :], x[:, None, :], dim=-1)
        return xcs, target

    def single_pos(self, xcs, target):
        target = torch.argmax(target, 1)
        loss = F.cross_entropy(xcs / self.temperature, target, reduction="mean")
        return loss

    def multiple_pos(self, xcs, target):
        # Standard binary cross entropy loss. We use binary_cross_entropy() and
        # not binary_cross_entropy_with_logits() because of
        # https://github.com/pytorch/pytorch/issues/102894.
        # The method *_with_logits() uses the log-sum-exp-trick, which causes
        # inf and -inf values to result in a NaN result.
        loss = F.binary_cross_entropy(
            (xcs / self.temperature).sigmoid(), target, reduction="none"
        )

        target_pos = target.bool()
        target_neg = ~target_pos

        loss_pos = loss * target_pos
        loss_neg = loss * target_neg

        loss_pos = loss_pos.sum(dim=1)
        loss_neg = loss_neg.sum(dim=1)
        num_pos = target.sum(dim=1)
        num_neg = xcs.size(0) - num_pos
        return ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()

    def forward(self, x, y, mask):
        # x: [B, Ch, H, W, D] in [-inf, inf]
        # y: [B, Classes, H, W, D] in [-1, 0, 1]
        # mask: [B, Classes, H, W, D] in [False, True]

        # eliminate classes with no annotations in the image
        non_empty = (y * mask).flatten(2).sum(-1).sum(0)  # sum spatial and batch
        non_empty = torch.nonzero(non_empty)  # [nClasses, 1]
        if len(non_empty) == 0:
            # if all bg, just skip it
            return torch.tensor(0).to(y)

        y = y[:, non_empty].squeeze_(2)  # [B, nClasses, H, D, W]
        mask = mask[:, non_empty].squeeze_(2)  # [B, nClasses, H, D, W]

        # TODO: l2 norm

        xcs, target = self.similarity(x, y, mask)

        return self.loss(xcs, target)


class BinaryDiceLoss_batch(nn.Module):
    def __init__(self, smooth=1, p=2, reduction="mean"):
        super(BinaryDiceLoss_batch, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, mask):
        # Inputs are in batch mode: [B, C, H, W, D]

        predict = F.sigmoid(predict).contiguous().flatten(2)
        target = target.contiguous().flatten(2)
        mask = mask.contiguous().flatten(2)

        # Standard Dice score
        num = (torch.mul(predict, target) * mask.int()).sum(2)
        den = ((predict + target) * mask.int()).sum(2)
        dice_score = 2 * num / (den + self.smooth)

        # Logits average
        pred_avg = predict.sum(-1) / mask.sum(-1)

        # Avoid overpenalization of categories not present in the batch
        # present: dice loss; not present: logits avg
        n_mask = torch.tensor(target.sum(-1) == 0, dtype=bool)  # negatives
        dice_loss = (1 - dice_score) * ~n_mask + (pred_avg * n_mask)

        return dice_loss.mean()


class SegmentationLoss_batch(nn.Module):
    def __init__(
        self,
        w_bce=1,
        w_dice=1,
        w_xent=0.01,
        prototypes=True,
        multiple_pos=True,
        entities=False,
    ) -> None:
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_xent = w_xent
        self.names_losses = {}

        if self.w_bce > 0:
            self.bce = nn.BCEWithLogitsLoss(reduction="none")
            self.names_losses["bce"] = w_bce

        if self.w_dice > 0:
            self.dice = BinaryDiceLoss_batch()
            self.names_losses["dice"] = w_dice

        if self.w_xent > 0:
            self.contrastive = ntBXentLoss(
                temperature=0.1,
                batch_size=16384,  # 4096
                prototypes=prototypes,
                multiple_pos=multiple_pos,
            )
            self.names_losses["xent"] = w_xent

        self.forward = self.forward_normal
        if entities:
            self.forward = self.forward_entities

    def forward_normal(self, predict, target, features, name):
        # Check datasets and target organs
        if isinstance(predict, list):
            predict = torch.stack(predict, 0)
        template_key = get_key(name[0])
        organ_list = TEMPLATE[template_key]

        idxs_o = torch.tensor(organ_list) - 1  # organ's indexes

        # mask to ignore regions if there are annotations discrepancies
        mask = target != -1

        # select target organs
        predict_o = predict[:, idxs_o]
        target_o = target[:, idxs_o]
        mask_o = mask[:, idxs_o]

        losses = dict.fromkeys(self.names_losses.keys(), 0)

        # binary cross-entropy loss
        if self.w_bce > 0:
            # bce = self.bce(predict_o, target_o)
            bce = sigmoid_focal_loss(predict_o, target_o)
            bce = torch.masked_select(bce, mask_o).mean()
            losses["bce"] = bce

        # soft-Dice loss
        if self.w_dice > 0:
            dice = self.dice(predict_o, target_o, mask_o)
            losses["dice"] = dice

        # contrastive loss (NT-xent with multiple positives)
        if self.w_xent > 0:
            xent = self.contrastive(features, target_o, mask_o)
            losses["xent"] = xent

        return losses

    def forward_entities(self, predict, target, features, name):
        # mask to ignore regions if there are annotations discrepancies
        mask = target != -1
        losses = dict.fromkeys(self.names_losses.keys(), 0)
        for B, nm in enumerate(name):
            key = get_key(nm)
            idxs_o = torch.tensor(TEMPLATE[key]) - 1

            target_o = target[B, idxs_o]
            non_empty_target = (target_o.flatten(1) > 0).sum(-1) > 0
            if non_empty_target.sum() == 0:
                non_empty_target = ~non_empty_target

            target_o = target_o[non_empty_target].unsqueeze(0)
            predict_o = predict[B][non_empty_target].unsqueeze(0)
            mask_o = mask[B, idxs_o][non_empty_target].unsqueeze(0)

            # binary cross-entropy loss
            if self.w_bce > 0:
                # bce = self.bce(predict_o, target_o)
                bce = sigmoid_focal_loss(predict_o, target_o)
                bce = torch.masked_select(bce, mask_o).mean()
                losses["bce"] += bce / len(name)

            # soft-Dice loss
            if self.w_dice > 0:
                dice = self.dice(predict_o, target_o, mask_o)
                losses["dice"] += dice / len(name)

            # contrastive loss (NT-xent with multiple positives)
            if self.w_xent > 0:
                xent = self.contrastive(features, target_o, mask_o)
                losses["xent"] += xent / len(name)

        return losses
