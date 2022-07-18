#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2022 Artem Moskalev
# Copyright (c) 2020 Massimiliano Patacchiola
# Copyright (c) 2020 Kris Korrel
# Paper: "Contrasting quadratic assignments for set-based representation learning", A. Moskalev & I. Sosnovik & V. Fischer & A. Smeulders, ECCV 2022
# GitHub: https://github.com/amoskalev/contrasting_quadratic
#
# Implementation of the SparceCLR contrastive objective
#
# Sparsemax/SparseMax implementation is based on https://github.com/KrisKorrel/sparsemax-pytorch

import math
import time
import collections

from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch import nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from utils import AverageMeter

def mine_positives(x):
    
    B = x.shape[0]
    assert B % 2 == 0, "takes only even batch_size"
    
    # mask for positives
    m = torch.zeros_like(x)
    d = torch.eye(B).bool().to(x.device)
    
    m[::2][d[::2]] = 1
    mask1 = torch.roll(m, shifts=1, dims=1)
    mask2 = torch.roll(m, shifts=1, dims=0)
    mask = mask1 + mask2
    
    # indexed pos
    pos = x[mask.bool()]
    
    return pos

def mine_negatives(x):
    
    B = x.shape[0]
    assert B % 2 == 0, "takes only even batch_size"
    
    # mask for negatives
    mask_neg = torch.ones_like(x) - torch.eye(B).to(x.device)
    
    # indexed neg
    num_negatives = B - 1
    
    neg = x[mask_neg.bool()].reshape(B, num_negatives)
    
    return neg

def add_margin(x, margin=1):
    
    B = x.shape[0]
    assert B % 2 == 0, "takes only even batch_size"
    
    # mask for positives
    m = torch.zeros_like(x)
    d = torch.eye(B).bool().to(x.device)
    
    m[::2][d[::2]] = 1
    mask1 = torch.roll(m, shifts=1, dims=1)
    mask2 = torch.roll(m, shifts=1, dims=0)
    mask = mask1 + mask2
    
    # add margin
    x_margin = x - margin*mask
    
    return x_margin

class sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        device = input.device
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

class Model(torch.nn.Module):
    def __init__(self, feature_extractor, out_dim=64):
        super(Model, self).__init__()
        
        self.out_dim = out_dim
        
        self.net = nn.Sequential(collections.OrderedDict([
          ("feature_extractor", feature_extractor)
        ]))

        self.head = nn.Sequential(collections.OrderedDict([
          ("linear1",  nn.Linear(feature_extractor.feature_size, 256)),
          ("bn1",      nn.BatchNorm1d(256)),
          ("relu",     nn.LeakyReLU()),
          ("linear2",  nn.Linear(256, self.out_dim)),
        ]))

        self.optimizer = Adam([{"params": self.net.parameters(), "lr": 0.001},
                               {"params": self.head.parameters(), "lr": 0.001}])
        
        self.sparse_max = sparsemax(dim=1)
        self.margin = 1
    
    def return_loss_fn(self, x, eps=1e-8):
        
        # compute cosine similarities
        n = torch.norm(x, p=2, dim=1, keepdim=True)
        x_pre = (x @ x.t()) / (n * n.t()).clamp(min=eps)
        
        # add margin
        x = add_margin(x_pre, margin=self.margin)
        
        #add positives & negatimes
        positives = mine_positives(x)
        negatives = mine_negatives(x)
        
        # compute SparseMAP objective
        sparsemax_arg = self.sparse_max(negatives).detach()
        support = torch.ne(sparsemax_arg, 0).long()
        
        supp_neg = support*negatives
        
        thresholding_tau_num = supp_neg.sum(-1) - 1
        support_norm = 1/(support.sum(-1))
        thresholding_tau = thresholding_tau_num*support_norm
        
        thr_tau_expanded = thresholding_tau.unsqueeze(-1).repeat(1, negatives.shape[-1])
        supp_thr = support*thr_tau_expanded
        
        sparse_map = 0.5*(supp_neg**2 - supp_thr**2).sum(-1) - positives + 0.5
        return sparse_map.mean()

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.net.train()
        self.head.train()
        loss_meter = AverageMeter()
        statistics_dict = {}
        for i, (data, data_augmented, _) in enumerate(train_loader):
                        
            data = torch.stack(data_augmented, dim=1)
            d = data.size()
            train_x = data.view(d[0]*2, d[2], d[3], d[4]).cuda()
            
            self.optimizer.zero_grad()                     
            features = self.net(train_x)
            tot_pairs = int(features.shape[0]*features.shape[0])            
            
            embeddings = self.head(features)
            loss = self.return_loss_fn(embeddings)
            loss_meter.update(loss.item(), features.shape[0])
            loss.backward()
            self.optimizer.step()
            if(i==0):
                statistics_dict["batch_size"] = data.shape[0]
                statistics_dict["tot_pairs"] = tot_pairs

        elapsed_time = time.time() - start_time 
        print("Epoch [" + str(epoch) + "]"
               + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
               + " loss: " + str(loss_meter.avg)
               + "; batch-size: " + str(statistics_dict["batch_size"])
               + "; tot-pairs: " + str(statistics_dict["tot_pairs"]))
                             
        return loss_meter.avg, -loss_meter.avg

    def save(self, file_path="./checkpoint.dat"):
        feature_extractor_state_dict = self.net.feature_extractor.state_dict()
        head_state_dict = self.head.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"backbone": feature_extractor_state_dict,
                    "head": head_state_dict,
                    "optimizer": optimizer_state_dict}, 
                    file_path)
        
    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.net.feature_extractor.load_state_dict(checkpoint["backbone"])
        self.head.load_state_dict(checkpoint["head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
