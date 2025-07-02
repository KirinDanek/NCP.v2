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
from AugmentedVGG16 import *
from prune_layer import *
import data as dataset

from operator import itemgetter
from heapq import nsmallest
import os
### the layers we disallow pruning on (ie before[-1], encode, decode)
DISALLOWED_LAYERS = {23, 24, 21} # debug: double check indices

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
        for module in self.model.before:
            module.register_forward_hook(fhook)
        if self.model.augmented:
            self.model.encode.register_forward_hook(fhook)
            self.model.decode.register_forward_hook(fhook)
        for module in self.model.after:
            module.register_forward_hook(fhook)
        for module in self.model.classifier:
            module.register_forward_hook(fhook)


    ### do forward pass and prepare for LRP. will build separate forward
    # funcs for gradient, etc
    def forward_lrp(self, x):
        in_size = x.size(0) ## batch size
        self.activation_to_layer = {} # activation index to actual conv layer
        self.grad_index = 0

        self.activation_index = 0 ## counts only conv layers
        for layer, module in enumerate(self.model.before):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                self.activation_to_layer[self.activation_index] = layer
                self.activation_index += 1
        layer_offset = len(self.model.before)

        #enc and decode if augmented
        # not prunable
        if self.model.augmented:
            x = self.model.encode(x)
            x = self.model.decode(x)
            layer_offset +=2
        
        for layer, module in enumerate(self.model.after):
            x = module(x)
            if isinstance(module, nn.Conv2d):
                self.activation_to_layer[self.activation_index] = layer_offset + layer
                self.activation_index += 1
        #flatten and classify
        x = x.view(in_size, -1)
        return self.model.classifier(x)
    
    #### DEBUG: ONLY BUILT TO RUN ON GAMMA HEURISTIC OR POSITIVE RELEVANCE BACKPROP
    def backward_lrp(self, R, relevance_method='z', param=1):
        modules = list(self.model.before)
        if self.model.augmented:
            modules += [self.model.encode, self.model.decode]
        modules += list(self.model.after)
        modules += list(self.model.classifier)
        modules = list(reversed(modules))

        if relevance_method == 'gamma' and param == 'heuristic': 
            # get gamma heuristic value from reverse index
            def get_augmented_vgg16_lrp_param(module_idx: int) -> float:
                """
                γ-schedule for LRP-γ on AugmentedVGG16, counting *from the output side* as we
                iterate through reversed(modules).

                ── classifier head ─────────────── 0.00
                ── Conv5 block  ─────────────────  0.00
                ── Augmented 1×1 + Conv4 block ─  0.10
                ── Conv3 block  ─────────────────  0.25
                ── Conv2 + Conv1 blocks ─────────  0.50  (all remaining layers)
                """
                if module_idx <= 6:                         # classifier layers
                    return 0.01 # 0.0
                elif 7 <= module_idx <= 13:                 # Conv5
                    return 0.01 # 0.0
                elif 14 <= module_idx <= 22:                # 1×1 augmented + Conv4
                    if module_idx == 15 or module_idx == 16: # augmented
                        return 0.00 
                    return 0.10 # 0.10
                elif 23 <= module_idx <= 29:                # Conv3
                    return 0.25 # 0.25
                else:     
                    if module_idx < 30 or module_idx > 39:
                        print(f'unexpected module index {module_idx}') 
                                                        # Conv2, Conv1, and anything earlier
                    return 0.50
            
            for i, module in enumerate(modules):
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
                if i == len(modules)-1:
                    R = lrp(module, R.data, lrp_var='first')
                else:    
                    dynamic_param = get_augmented_vgg16_lrp_param(i)
                    R = lrp(module, R.data, lrp_var=relevance_method, param=dynamic_param)

        elif relevance_method == 'z' and param == 1: ### POSITIVE RELEVANCE ONLY
            for i, module in enumerate(modules):
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
                if i == len(modules)-1:
                    R = lrp(module, R.data, lrp_var='first')
                else: 
                    R = lrp(module, R.data, lrp_var=relevance_method, param=param)
        else: 
            raise NotImplementedError

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            if self.args.relevance: ### if LRP, avg over trials (not normalize)
                v = self.filter_ranks[i]
                v = v / torch.sum(v) #  / num(dataset)
                self.filter_ranks[i] = v.cpu()
    
    # re index the filters to prune so that indices align during pruning
    # return list of (layer_num, filter_num)
    def get_pruning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        #filters_to_prune: filters to be pruned 1) layer number, 2) filter number, 3) its value

        # after each of the k filters are pruned, 
        # the filter index of the next filters change since the model is smaller
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            ## filter at index idx shifts down by however many filters
            # have already been pruned (i)
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune ## =list of (layer_num, filter_num)
    # num: number of filters to prune
    def lowest_ranking_filters(self, num):
        data = []
        if self.model.augmented == False:
            raise NotImplementedError
        for i in sorted(self.filter_ranks.keys()):
            layer_idx = self.activation_to_layer[i]
            if layer_idx in DISALLOWED_LAYERS: #debug: skipping disallowed conv layers
                continue
            for j in range(self.filter_ranks[i].size(0)):
                #(layer idx, filter idx, score)
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        # return num tuples w/ smallest relevance score
        return nsmallest(num, data, itemgetter(2)) 
    
class PruningFineTuner:
    def __init__(self, args, model):
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        self.args = args
        self.setup_dataloaders()
        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.model, args)
        self.model.train()
        self.save_loss = True

    def setup_dataloaders(self):
        from torchvision import datasets, transforms
        kwargs = {'num_workers': 0, 'pin_memory': True} if self.args.cuda else {}
        
        # Data Acquisition
        get_dataset = {
            #"cifar10": dataset.get_cifar10,  # CIFAR-10
            #'imagenet': dataset.get_imagenet, # ImageNet
            'basketball_imagenet': dataset.get_basketball_imagenet
        }[self.args.data_type.lower()]
        train_dataset, test_dataset = get_dataset()
        print(f"train_dataset:{len(train_dataset)}, test_dataset:{len(test_dataset)}")
        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=self.args.train_batch_size,
                                                        shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=self.args.test_batch_size,
                                                    shuffle=False, **kwargs)

        self.train_num = len(self.train_loader)
        self.test_num = len(self.test_loader)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        for data, target in self.test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = self.model(data)

            test_loss += self.criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        # self.correct += correct

        if self.save_loss:
            self.test_acc_tot.append((100. * correct).numpy() / len(self.test_loader.dataset))
            self.test_loss_tot.append(test_loss)
            self.test_iter.append(self.niter)

            # FLOP calculation
            #sample_batch = torch.FloatTensor(1, 3, 224, 224).cuda()
            #self.model = add_flops_counting_methods(self.model)
            #self.model.eval().start_flops_count()
            #_ = self.model(sample_batch)
            #self.flop_val.append(flops_to_string(self.model.compute_average_flops_cost()))
            #self.num_param.append(get_model_parameters_number(self.model))
            #print('Flops:  {}'.format(flops_to_string(self.model.compute_average_flops_cost())))
            #print('Params: ' + get_model_parameters_number(self.model))

        self.model.train()

    def train_epoch(self, optimizer=None, rank_filters=False):
        self.train_loss_batch = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.train_batch(optimizer, batch_idx, data, target, rank_filters)

        if self.save_loss:
            self.train_loss_tot.append(self.train_loss_batch / len(self.train_loader.dataset))

    def train_batch(self, optimizer, batch_idx, batch, label, rank_filters):
        self.model.zero_grad()

        if rank_filters:
            if self.args.relevance:  # lrp_based
                output = self.pruner.forward_lrp(batch)

                T = torch.zeros_like(output)
                for ii in range(len(label)):
                    T[ii,label[ii]] = 1.0
                # debug: uncomment ^ and comment v if not pruning wrt target class only
                #T[:, self.args.target_class_idx] = 1.0 

                self.pruner.backward_lrp(T.data)

                print('Train Epoch: [{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(batch), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader)))
                loss = self.criterion(output, label)
                self.train_loss_batch += loss.item()

            else:  # gradient_based
                raise NotImplementedError
                output = self.pruner.forward(batch)
                loss = self.criterion(output, label)
                loss.backward()
                print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(batch), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
                self.train_loss_batch += loss.item()
        else:
            loss = self.criterion(self.model(batch), label)
            loss.backward()
            optimizer.step()
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(batch), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))
            self.train_loss_batch += loss.item()

    def total_num_filters(self):
        # Conv layer의 모든 filter 수를 counting
        filters = 0
        for module in self.model.before: #debug: don't count filters from encode and decode
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters += module.out_channels
        for module in self.model.after: 
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters += module.out_channels
        return filters

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(self.model.classifier.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # self.correct = 0
        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning")
        # self.correct /= epoches

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.pruner.reset()

        self.train_epoch(rank_filters=True)
        # training 하면서 동시에 hook 써서 후보 찾기 #
        # (각 layer 마다 compute_rank 안에서 계산되어서 self.filter_ranks list에 저장이 된다.

        self.pruner.normalize_ranks_per_layer()  # Normalization 

        return self.pruner.get_pruning_plan(num_filters_to_prune) 

    def forward_hook(self):
        # Handle either before/after or features depending on model state
        for module in self.model.before:
            module.register_forward_hook(fhook)
        if self.model.augmented:
            self.model.encode.register_forward_hook(fhook)
            self.model.decode.register_forward_hook(fhook)
            
        for module in self.model.after:
                module.register_forward_hook(fhook)

        for module in self.model.classifier:
            module.register_forward_hook(fhook)


    def prune(self):
        self.train_loss_tot = []
        self.test_loss_tot = []
        self.test_acc_tot = []
        self.test_iter = []
        #self.flop_val =[] # debug: commented out
        self.num_param = []
        #self.R_tot = [] # debug: commented out
        self.data_tot = []
        self.time_tot = []
        self.save_loss = True

        # Get the accuracy before pruning
        self.niter = 0
        self.temp = 0
        self.test()
        self.model.train()

        # Make sure all the layers are trainable except for augmented layers
        for param in self.model.before.parameters():
            param.requires_grad = True
        if self.model.augmented:
            for param in self.model.encode.parameters():
                param.requires_grad = False
            for param in self.model.decode.parameters():
                param.requires_grad = False
        for param in self.model.after.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = int(number_of_filters * self.args.pr_step)  # 0.05 (5%) -> 0.01 (1%) temporally
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        iterations = int(iterations * self.args.total_pr) #up to 80%

        print(f"Number of pruning iterations to reduce {self.args.total_pr}% filters: ", iterations)

        #R_tot, data_tot, time_tot = self.lrp()  # lrp using conventional model # debug: removed these two lines
        #self.R_tot.append(R_tot)

        for kk in range(iterations):
            print("Ranking filters.. {}".format(kk))
            self.niter += 1
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration) 
            # prune_targets: 잘라야 할 filter들의 1) layer number, 2) filter number가 넘어옴
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] += 1

            print("Layers that will be pruned", layers_prunned)  # 총 잘릴 layer 별 filter 수
            print("Pruning filters.. ")
            model = self.model.cpu()  # 현재 모델 갖다가..
            for layer_index, filter_index in prune_targets:  # 하나씩 꺼내서 자르기 시작
                model = prune_conv_layer_sequential(model, layer_index, filter_index, cuda_flag=self.args.cuda)

            self.model = model.cuda() if self.args.cuda else model

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            ### note: this is calculating the pct of remaining filters
            # (not pct of pruned filters)
            print("Filters pruned", str(message))
            self.test()  # 잘리고 나서 test 해봄
            print("Fine tuning to recover from pruning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum) 
            self.train(optimizer, epoches=10)
            #R_tot, data_tot, time_tot = self.lrp() # debug: removed these 3 lines
            #self.R_tot.append(R_tot)
            #del R_tot

        print("Finished. Removing augmented layers and fine tuning")
        #removing augmented layers in forward pass
        self.model.augmented = False
        self.niter += 1

        ## remove encode and decode before final fine tuning
        del self.model.encode
        del self.model.decode

        # final fine tuning on all layers
        for param in self.model.before.parameters():
            param.requires_grad = True
        for param in self.model.after.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        self.train(optimizer, epoches=15)

