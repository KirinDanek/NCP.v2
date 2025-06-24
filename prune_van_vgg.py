### 

import numpy as np
import torch
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.autograd import Variable, Function

from lrp import *
from prune_layer import *

from operator import itemgetter
from heapq import nsmallest
import os


def fhook(self, input, output):
    self.input = input[0]
    self.output = output.data

class FilterPruner:
    def __init__(self, model, args):
        self.model = model
        self.reset()
        self.args = args

    def reset(self):
        self.filter_ranks = {}
        self.forward_hook()

    def forward_hook(self):
        for name, module in self.model.features._modules.items():
            module.register_forward_hook(fhook)
        for name, module in self.model.classifier._modules.items():
            module.register_forward_hook(fhook)

    ### do forward pass and prepare for backprop
    def forward_lrp(self, x):
        in_size = x.size(0) ## batch size
        self.activation_to_layer = {} # activation index to actual conv layer
        self.grad_index = 0

        self.activation_index = 0 ## counts only conv layers
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                self.activation_to_layer[self.activation_index] = layer
                self.activation_index += 1
                
        x = x.view(in_size, -1)
        return self.model.classifier(x)
    
    #### DEBUG: ONLY BUILT TO RUN ON GAMMA HEURISTIC OR POSITIVE RELEVANCE BACKPROP
    def backward_lrp(self, R, relevance_method='z', param=1):
    
        if relevance_method == 'gamma' and param == 'heuristic': ## todo: sum absolute vals of relevance for pruning
            # get gamma heuristic value from reverse index
            def get_vgg16_lrp_param(module_idx: int) -> float:
                if module_idx <= 6:                         # classifier layers
                    return 0.0 # 0.0
                elif 7 <= module_idx <= 13:                 # Conv5
                    return 0.0 #0.0
                elif 14 <= module_idx <= 20:                # 1×1 augmented + Conv4
                    return 0.10 # 0.10
                elif 21 <= module_idx <= 27:                # Conv3
                    return 0.25 # 0.25
                elif module_idx < 28 or module_idx > 37:
                    print(f'unexpected module index {module_idx}')                                       
                return 0.50 # Conv2, Conv1
            
            for i, module in enumerate(reversed(list(self.model.features) + list(self.model.classifier))):
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    activation_index = self.activation_index - self.grad_index - 1
                    ### summing over batch + spatial dims (per-filter relevance)
                    ## shape (num_filters,)
                    values = R.abs().sum(dim=(0,2,3)).data # debug: absolute relevance

                    if activation_index not in self.filter_ranks:
                        self.filter_ranks[activation_index] = torch.FloatTensor(
                            R.size(1)).zero_().cuda() if self.args.cuda else torch.FloatTensor(
                            R.size(1)).zero_()
                    ## add batch scores to total
                    self.filter_ranks[activation_index] += values
                    self.grad_index += 1
                if i == 37:
                    R = lrp(module, R.data, lrp_var='first')
                else:    
                    dynamic_param = get_vgg16_lrp_param(i)
                    R = lrp(module, R.data, lrp_var=relevance_method, param=dynamic_param)

        else: ### POSITIVE RELEVANCE ONLY
            for i, module in enumerate(reversed(list(self.model.features) + list(self.model.classifier))):
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    activation_index = self.activation_index - self.grad_index - 1
                    ### summing over batch + spatial dims (per-filter relevance)
                    ## shape (num_filters,)
                    values = torch.sum(R, dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data

                    if activation_index not in self.filter_ranks:
                        self.filter_ranks[activation_index] = torch.FloatTensor(
                            R.size(1)).zero_().cuda() if self.args.cuda else torch.FloatTensor(
                            R.size(1)).zero_()
                    ## add batch scores to total
                    self.filter_ranks[activation_index] += values
                    self.grad_index += 1
                if i == 37: ### debug: hardcoded first conv layer index
                    R = lrp(module, R.data, lrp_var='first')
                else: 
                    R = lrp(module, R.data, lrp_var=relevance_method, param=param)

    def forward(self, x):
        self.activations = []
        self.weights = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)

                #### TODO: make sure hooks are being computed 
                ## compatibly with LRP implementation