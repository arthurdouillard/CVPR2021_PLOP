import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(loss_type):
    if loss_type == 'focal_loss':
        return FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


def soft_crossentropy(logits, labels, logits_old, mask_valid_pseudo,
                      mask_background, pseudo_soft, pseudo_soft_factor=1.0):
    if pseudo_soft not in ("soft_certain", "soft_uncertain"):
        raise ValueError(f"Invalid pseudo_soft={pseudo_soft}")
    nb_old_classes = logits_old.shape[1]
    bs, nb_new_classes, w, h = logits.shape

    loss_certain = F.cross_entropy(logits, labels, reduction="none", ignore_index=255)
    loss_uncertain = (torch.log_softmax(logits_old, dim=1) * torch.softmax(logits[:, :nb_old_classes], dim=1)).sum(dim=1)

    if pseudo_soft == "soft_certain":
        mask_certain = ~mask_background
        mask_uncertain = mask_valid_pseudo & mask_background
    elif pseudo_soft == "soft_uncertain":
        mask_certain = (mask_valid_pseudo & mask_background) | (~mask_background)
        mask_uncertain = ~mask_valid_pseudo & mask_background

    loss_certain = mask_certain.float() * loss_certain
    loss_uncertain = (~mask_certain).float() * loss_uncertain

    return loss_certain + pseudo_soft_factor * loss_uncertain


class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction="mean", ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class FocalLossNew(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction="mean", ignore_index=255, index=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.index = index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        mask_new = (targets >= self.index).float()
        focal_loss = mask_new * focal_loss + (1. - mask_new) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class BCEWithLogitsLossWithIgnoreIndex(nn.Module):

    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)


class IcarlLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=255, bkg=False):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.bkg = bkg

    def forward(self, inputs, targets, output_old):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        if self.bkg:
            targets[:, 1:output_old.shape[1], :, :] = output_old[:, 1:, :, :]
        else:
            targets[:, :output_old.shape[1], :, :] = output_old

        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UnbiasedCrossEntropy(nn.Module):

    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets, mask=None):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        # Following line was fixed more recently in:
        # https://github.com/fcdl94/MiB/commit/1c589833ce5c1a7446469d4602ceab2cdeac1b0e
        # and added to my repo the 04 August 2020 at 10PM
        labels = targets.clone()  # B, H, W

        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        if mask is not None:
            labels[mask] = self.ignore_index
        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


def nca(
    similarities,
    targets,
    loss,
    class_weights=None,
    focal_gamma=None,
    scale=1,
    margin=0.,
    exclude_pos_denominator=True,
    hinge_proxynca=False,
    memory_flags=None,
):
    """Compute AMS cross-entropy loss.

    Reference:
        * Goldberger et al.
          Neighbourhood components analysis.
          NeuriPS 2005.
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.

    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate
                         anything else.
    :return: A float scalar loss.
    """
    b = similarities.shape[0]
    c = similarities.shape[1]
    w = similarities.shape[-1]

    if margin > 0.:
        similarities = similarities.view(b, c, w * w)
        targets = targets.view(b * w * w)
        margins = torch.zeros_like(similarities)
        margins = margins.permute(0, 2, 1)
        margins[torch.arange(margins.shape[0]), targets, :] = margin
        margins = margins.permute(0, 2, 1)
        similarities = similarities - margin
        similarities = similarities.view(b, c, w, w)
        targets = targets.view(b, w, w)

    similarities = scale * similarities

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(dim=1, keepdims=True)[0]  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return loss(similarities, targets)


class NCA(nn.Module):

    def __init__(self, scale=1., margin=0., ignore_index=255, reduction="mean"):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.scale = scale
        self.margin = margin

    def forward(self, inputs, targets):
        return nca(inputs, targets, self.ce, scale=self.scale, margin=self.margin)


class UnbiasedNCA(nn.Module):

    def __init__(self, scale=1., margin=0., old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.unce = UnbiasedCrossEntropy(old_cl, reduction, ignore_index)
        self.scale = scale
        self.margin = margin

    def forward(self, inputs, targets):
        return nca(inputs, targets, self.unce, scale=self.scale, margin=self.margin)


class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1., kd_cil_weights=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kd_cil_weights = kd_cil_weights

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)
        if self.kd_cil_weights:
            w = -(torch.softmax(targets, dim=1) * torch.log_softmax(targets, dim=1)).sum(dim=1) + 1.0
            loss = loss * w[:, None]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs

class ExcludedKnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', index_new=-1, new_reduction="gt",
                 initial_nb_classes=-1, temperature_semiold=1.0):
        super().__init__()
        self.reduction = reduction

        self.initial_nb_classes = initial_nb_classes
        self.temperature_semiold = temperature_semiold

        #assert index_new > 0, index_new
        self.index_new = index_new
        if new_reduction not in ("gt", "sum"):
            raise ValueError(f"Unknown new_reduction={new_reduction}")
        self.new_reduction = new_reduction

    def forward(self, inputs, targets, labels, mask=None):
        bs, ch_new, w, h = inputs.shape
        device = inputs.device
        labels_no_unknown = labels.clone()
        labels_no_unknown[labels_no_unknown == 255] = 0

        temperature_semiold = torch.ones(bs, self.index_new + 1, w , h).to(device)
        if self.index_new > self.initial_nb_classes:
            temperature_semiold[:, self.initial_nb_classes:self.index_new] = temperature_semiold[:, self.initial_nb_classes:self.index_new] / self.temperature_semiold

        # 1. If pixel is from new class
        new_inputs = torch.zeros(bs, self.index_new + 1, w, h).to(device)
        new_targets = torch.zeros(bs, self.index_new + 1, w, h).to(device)

        #   1.1. new_bg -> 0
        new_targets[:, 0] = 0.
        new_inputs[:, 0] = inputs[:, 0]
        #   1.2. new_old -> old_old
        new_targets[:, 1:self.index_new] = targets[:, 1:]
        new_inputs[:, 1:self.index_new] = inputs[:, 1:self.index_new]
        #   1.3. new_new GT -> old_bg
        if self.new_reduction == "gt":
            nb_pixels = bs * w * h
            new_targets[:, self.index_new] = targets[:, 0]
            tmp = inputs.view(bs, ch_new, w * h).permute(0, 2, 1).reshape(nb_pixels, ch_new)[torch.arange(nb_pixels), labels_no_unknown.view(nb_pixels)]
            tmp = tmp.view(bs, w, h)
            new_inputs[:, self.index_new] = tmp
        elif self.new_reduction == "sum":
            new_inputs[:, self.index_new] = inputs[:, self.index_new:].sum(dim=1)

        loss_new = -(torch.log_softmax(temperature_semiold * new_inputs, dim=1) * torch.softmax(temperature_semiold * new_targets, dim=1)).sum(dim=1)

        # 2. If pixel is from old class
        old_inputs = torch.zeros(bs, self.index_new + 1, w, h).to(device)
        old_targets = torch.zeros(bs, self.index_new + 1, w, h).to(device)

        #   2.1. new_bg -> old_bg
        old_targets[:, 0] = targets[:, 0]
        old_inputs[:, 0] = inputs[:, 0]
        #   2.2. new_old -> old_old
        old_targets[:, 1:self.index_new] = targets[:, 1:self.index_new]
        old_inputs[:, 1:self.index_new] = inputs[:, 1:self.index_new]
        #   2.3. new_new -> 0
        if self.new_reduction == "gt":
            old_targets[:, self.index_new] = 0.
            tmp = inputs.view(bs, ch_new, w * h).permute(0, 2, 1).reshape(nb_pixels, ch_new)[torch.arange(nb_pixels), labels_no_unknown.view(nb_pixels)]
            tmp = tmp.view(bs, w, h)
            old_inputs[:, self.index_new] = tmp
        elif self.new_reduction == "sum":
            old_inputs[:, self.index_new] = inputs[:, self.index_new:].sum(dim=1)

        loss_old = -(torch.log_softmax(temperature_semiold * old_inputs, dim=1) * torch.softmax(temperature_semiold * old_targets, dim=1)).sum(dim=1)

        mask_new = (labels >= self.index_new) & (labels < 255)
        mask_old = labels < self.index_new
        loss = (mask_new.float() * loss_new) + (mask_old.float() * loss_old)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss


class BCESigmoid(nn.Module):
    def __init__(self, reduction="mean", alpha=1.0, shape="trim"):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.shape = shape

    def forward(self, inputs, targets, mask=None):
        nb_old_classes = targets.shape[1]
        if self.shape == "trim":
            inputs = inputs[:, :nb_old_classes]
        elif self.shape == "sum":
            inputs[:, 0] = inputs[:, nb_old_classes:].sum(dim=1)
            inputs = inputs[:, :nb_old_classes]
        else:
            raise ValueError(f"Unknown parameter to handle shape = {self.shape}.")

        inputs = torch.sigmoid(self.alpha * inputs)
        targets = torch.sigmoid(self.alpha * targets)

        loss = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss


class UnbiasedKnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(
            inputs.device
        )

        den = torch.logsumexp(inputs, dim=1)  # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(
            torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1
        ) - den  # B, H, W

        labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg +
                (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs
