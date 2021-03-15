import copy
import math
import os
from functools import partial, reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init

import inplace_abn
import models
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync
from modules import DeeplabV3


def make_model(opts, classes=None):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abn':
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    else:
        norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex

    if opts.norm_act == "iabn_sync_test":
        opts.norm_act = "iabn_sync"

    body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
    if not opts.no_pretrained:
        pretrained_path = os.path.join(opts.code_directory, f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar')
        pre_dict = torch.load(pretrained_path, map_location='cpu')
        for key in copy.deepcopy(list(pre_dict['state_dict'].keys())):
            pre_dict['state_dict'][key[7:]] = pre_dict['state_dict'].pop(key)
        del pre_dict['state_dict']['classifier.fc.weight']
        del pre_dict['state_dict']['classifier.fc.bias']

        body.load_state_dict(pre_dict['state_dict'])
        del pre_dict  # free memory

    head_channels = 256

    head = DeeplabV3(
        body.out_channels,
        head_channels,
        256,
        norm_act=norm,
        out_stride=opts.output_stride,
        pooling_size=opts.pooling
    )

    if classes is not None:
        model = IncrementalSegmentationModule(
            body,
            head,
            head_channels,
            classes=classes,
            fusion_mode=opts.fusion_mode,
            nb_background_modes=opts.nb_background_modes,
            multimodal_fusion=opts.multimodal_fusion,
            use_cosine=opts.cosine,
            disable_background=opts.disable_background,
            only_base_weights=opts.base_weights,
            opts=opts
        )
    else:
        model = SegmentationModule(body, head, head_channels, opts.num_classes, opts.fusion_mode)

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationModule(nn.Module):

    def __init__(
        self,
        body,
        head,
        head_channels,
        classes,
        ncm=False,
        fusion_mode="mean",
        nb_background_modes=1,
        multimodal_fusion="sum",
        use_cosine=False,
        disable_background=False,
        only_base_weights=False,
        opts=None
    ):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"

        use_bias = not use_cosine

        if nb_background_modes > 1:
            classes[0] -= 1
            classes = [nb_background_modes] + classes

        if only_base_weights:
            classes = [classes[0]]

        if opts.dataset == "cityscapes_domain":
            classes = [opts.num_classes]

        self.cls = nn.ModuleList([nn.Conv2d(head_channels, c, 1, bias=use_bias) for c in classes])
        self.classes = classes
        self.head_channels = head_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None

        self.multi_modal_background = nb_background_modes > 1
        self.disable_background = disable_background

        self.nb_background_modes = nb_background_modes
        self.multimodal_fusion = multimodal_fusion

        self.use_cosine = use_cosine
        if use_cosine:
            self.scalar = nn.Parameter(torch.tensor(1.)).float()
            assert not self.multi_modal_background
        else:
            self.scalar = None

        self.in_eval = False

    def align_weight(self, align_type):
        old_weight_norm = self._compute_weights_norm(self.cls[:-1], only=align_type)

        new_weight_norm = self._compute_weights_norm(self.cls[-1:])

        gamma = old_weight_norm / new_weight_norm

        self.cls[-1].weight.data = gamma * self.cls[-1].weight.data

    def _compute_weights_norm(self, convs, only="all"):
        c = 0
        s = 0.

        for i, conv in enumerate(convs):
            w = conv.weight.data[..., 0, 0]

            if only == "old" and i == 0:
                w = w[1:]
            elif only == "background" and i == 0:
                w = w[:1]

            s += w.norm(dim=1).sum()
            c += w.shape[0]

        return s / c

    def _network(self, x, ret_intermediate=False, only_bg=False):
        x_b, attentions = self.body(x)
        x_pl = self.head(x_b)
        out = []

        if self.use_cosine:
            x_clf = x_pl.permute(0, 2, 3, 1)
            x_clf = x_clf.reshape(x_pl.shape[0] * x_pl.shape[2] * x_pl.shape[3], x_pl.shape[1])
            x_clf = F.normalize(x_clf, dim=1, p=2)
            x_clf = x_clf.view(x_pl.shape[0], x_pl.shape[2], x_pl.shape[3], x_pl.shape[1])
            x_clf = x_clf.permute(0, 3, 1, 2)
        else:
            x_clf = x_pl

        if only_bg:
            return self.cls[0](x_pl)
        else:
            for i, mod in enumerate(self.cls):
                if i == 0 and self.multi_modal_background:
                    out.append(self.fusion(mod(x_pl)))
                elif self.use_cosine:
                    w = F.normalize(mod.weight, dim=1, p=2)
                    out.append(F.conv2d(x_pl, w))
                else:
                    out.append(mod(x_pl))

        x_o = torch.cat(out, dim=1)

        if self.disable_background and self.in_eval:
            x_o[:, 0] = 0.

        if ret_intermediate:
            return x_o, x_b, x_pl, attentions
        return x_o

    def fusion(self, tensors):
        if self.multimodal_fusion == "sum":
            return tensors.sum(dim=1, keepdims=True)
        elif self.multimodal_fusion == "mean":
            return tensors.mean(dim=1, keepdims=True)
        elif self.multimodal_fusion == "max":
            return tensors.max(dim=1, keepdims=True)[0]
        elif self.multimodal_fusion == "softmax":
            return (F.softmax(tensors, dim=1) * tensors).sum(dim=1, keepdims=True)
        else:
            raise NotImplementedError(
                f"Unknown fusion mode for multi-modality: {self.multimodal_fusion}."
            )

    def init_new_classifier(self, device):
        cls = self.cls[-1]

        if self.multi_modal_background:
            imprinting_w = self.cls[0].weight.sum(dim=0)
            bkg_bias = self.cls[0].bias.sum(dim=0)
        else:
            imprinting_w = self.cls[0].weight[0]
            if not self.use_cosine:
                bkg_bias = self.cls[0].bias[0]

        if not self.use_cosine:
            bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)
            new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        if not self.use_cosine:
            cls.bias.data.copy_(new_bias)

        if self.multi_modal_background:
            self.cls[0].bias.data.copy_(new_bias.squeeze(0))
        else:
            if not self.use_cosine:
                self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def init_new_classifier_multimodal(self, device, train_loader, init_type):
        print("Init new multimodal classifier")
        winners = torch.zeros(self.nb_background_modes,
                              self.classes[-1]).to(device, dtype=torch.long)

        nb_old_classes = sum(self.classes[1:-1]) + 1

        for images, labels in train_loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            modalities = self.forward(images, only_bg=True)[0].argmax(dim=1)
            mask = (0 < labels) & (labels < 255)

            modalities = modalities[mask].view(-1)
            labels = labels[mask].view(-1)

            winners.index_put_(
                (modalities, labels - nb_old_classes),
                torch.LongTensor([1]).expand_as(modalities).to(device),
                accumulate=True
            )

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        if "_" in init_type:
            init_type, to_reinit = init_type.split("_")
        else:
            to_reinit = None

        for c in range(self.classes[-1]):
            if init_type == "max":
                modality = winners[:, c].argmax()
                new_weight = self.cls[0].weight.data[modality]
                new_bias = (self.cls[0].bias.data[modality] - bias_diff)[0]
            elif init_type == "softmax":
                modality = winners[:, c].argmax()
                weighting = F.softmax(winners[:, c].float(), dim=0)
                new_weight = (weighting[:, None, None, None] * self.cls[0].weight.data).sum(dim=0)
                new_bias = (weighting * self.cls[0].bias.data).sum(dim=0)
            else:
                raise ValueError(f"Unknown multimodal init type: {init_type}.")

            self.cls[-1].weight.data[c].copy_(new_weight)
            self.cls[-1].bias.data[c].copy_(new_bias)

            self.cls[0].bias.data[modality].copy_(new_bias)

            if to_reinit is not None:
                if to_reinit == "init":
                    init.kaiming_uniform_(self.cls[0].weights.data[modality], a=math.sqrt(5))
                    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(self.cls[0].bias.data[modality], -bound, bound)
                elif to_reinit == "remove":
                    self.cls[0].bias.data = torch.cat(
                        (self.cls[0].bias.data[:modality], self.cls[0].bias.data[modality + 1:])
                    )

    def forward(self, x, scales=None, do_flip=False, ret_intermediate=False, only_bg=False):
        out_size = x.shape[-2:]

        out = self._network(x, ret_intermediate, only_bg=only_bg)

        sem_logits_small = out[0] if ret_intermediate else out

        sem_logits = F.interpolate(
            sem_logits_small, size=out_size, mode="bilinear", align_corners=False
        )

        if ret_intermediate:
            return sem_logits, {
                "body": out[1],
                "pre_logits": out[2],
                "attentions": out[3] + [out[2]],
                "sem_logits_small": sem_logits_small
            }

        return sem_logits, {}

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
