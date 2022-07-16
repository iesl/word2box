import torch
import torch.nn as nn
import os
import json
import numpy as np


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path), map_location="cpu"))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def get_parameters(self, mode="numpy", param_dict=None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()


class MaskedAvgPoolingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def broadcast_mask(self, tensor, mask):
        if len(tensor.shape) == 4:
            mask.unsqueeze_(-1).unsqueeze_(-1)
        elif len(tensor.shape) == 3:
            mask.unsqueeze_(-1)
        else:
            raise ValueError("the tensor dimension for context is wrong")
        return mask

    def forward(self, tensor, mask, dim):
        """
        Tensor is any tensor
        Mask is a binary tensor which is 1 for any part of Tensor which should be included
        dim is dimension to take the mean over
        """
        mask = self.broadcast_mask(tensor, mask)
        return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)
