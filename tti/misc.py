"""
Miscellaneous Utility Functions
"""
import click
import warnings
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from copy import deepcopy

import random

def colorful_print(string: str, *args, **kwargs) -> None:
    print(click.style(string, *args, **kwargs))

def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs))
def get_image(image_path):
    with Image.open(image_path) as img:
        return img.convert("RGB")
        return np.array(img)

def pad_from_left(input_id_list, pad_token_id):
    max_len = max([len(input_id) for input_id in input_id_list])
    if len(input_id_list) == 1:
        max_len += random.randint(1, 100) # add some randomness to the padding if the batch size is one, for better batch inference
    padded_input_ids = [[pad_token_id]*(max_len-len(input_id)) + input_id for input_id in input_id_list]
    return padded_input_ids

def merge_dicts(dict_list):
    merged_dict = {}
    for dictionary in dict_list:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]
    return merged_dict