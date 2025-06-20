import copy
import itertools
import functools
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as torch_transforms
import encoding.datasets as enc_ds
from .lerf import LerfSegmentation

encoding_datasets = {
    x: functools.partial(enc_ds.get_dataset, x)
    for x in ["coco", "ade20k", "pascal_voc", "pascal_aug", "pcontext", "citys"]
}
def _lerf_loader(**kwargs):
    print("[DEBUG] _lerf_loader received kwargs:", kwargs)
    # Robustly prefer input_dir if present and not None/empty
    root = kwargs.get('input_dir', None)
    if root is None or root == '':
        root = kwargs.get('root', None)
    if root is None or root == '':
        raise ValueError("LERF loader requires 'input_dir' or 'root' argument.")
    # Remove input_dir/root from kwargs to avoid duplicate argument error
    kwargs = dict(kwargs)
    kwargs.pop('input_dir', None)
    kwargs.pop('root', None)
    return LerfSegmentation(root=root, **kwargs)
encoding_datasets["lerf"] = _lerf_loader


def get_custom_dataset(name, **kwargs):
    if name in encoding_datasets:
        return encoding_datasets[name.lower()](**kwargs)
    assert False, f"dataset {name} not found"


def get_available_datasets():
    return list(encoding_datasets.keys())

get_dataset = get_custom_dataset
