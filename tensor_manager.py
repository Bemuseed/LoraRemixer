import torch
import numpy as np

def get_tensors(dct):
    tensors = list(dct.values())
    new_dict = dict()
    for k in list(dct.keys()):
        new_dict[k] = 0
    return [np.atleast_1d(t.numpy()) for t in tensors], new_dict 

def restore_tensor_dict(tensors, dict_template):
    t_counter = 0
    new_dict = {}
    for k in list(dict_template.keys()):
       new_dict[k] = torch.from_numpy(tensors[t_counter])
       t_counter += 1
    return new_dict

def get_tensors_from_file(ckpt_file):
    tensor_dict = torch.load(ckpt_file, map_location="cpu")
    return get_tensors(tensor_dict)
