import torch
import torch.nn as nn

from .lseg_vit import (
    _make_pretrained_clip_vitl16_384,
    _make_pretrained_clip_vitb32_384,
    _make_pretrained_clip_vitb16_384,  # ensure this is imported
    _make_pretrained_clipRN50x16_vitl16_384,
    forward_vit,
)


def _make_encoder(
    backbone,
    features,
    use_pretrained=True,
    groups=1,
    expand=False,
    exportable=True,
    hooks=None,
    use_vit_only=False,
    use_readout="ignore",
    enable_attention_hooks=False,
):  
    if backbone == "clip_vitl16_384": 
        clip_pretrained, pretrained = _make_pretrained_clip_vitl16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        ) 
    elif backbone == "clipRN50x16_vitl16_384":
        clip_pretrained, pretrained = _make_pretrained_clipRN50x16_vitl16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )
    elif backbone == "clip_vitb32_384":
        clip_pretrained, pretrained = _make_pretrained_clip_vitb32_384(
            use_pretrained, 
            hooks=hooks, 
            use_readout=use_readout,
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        ) 
    elif backbone == "clip_vitb16_384":
        clip_pretrained, pretrained = _make_pretrained_clip_vitb16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return clip_pretrained, pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch