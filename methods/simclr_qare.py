#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2022 Artem Moskalev
# Copyright (c) 2020 Massimiliano Patacchiola
# Paper: "Contrasting quadratic assignments for set-based representation learning", A. Moskalev & I. Sosnovik & V. Fischer & A. Smeulders, ECCV 2022
# GitHub: https://github.com/amoskalev/contrasting_quadratic
#
# Implementation of our SimCLR + QARe described in the paper.
# The implementation of SimCLR is based on: https://github.com/pietz/simclr/blob/master/SimCLR.ipynb
# and: https://github.com/mpatacchiola/self-supervised-relational-reasoning

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

from .objectives import QARe

class Model(torch.nn.Module):
    def __init__(self, feature_extractor, out_dim=64, alpha=1, beta=1):
        super(Model, self).__init__()
        
        self.qap_objective = QARe()
        self.alpha = alpha
        self.beta = beta
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
            
    def get_similarities(self, in_x, in_y, eps):
        n_in_x = nn.functional.normalize(in_x, p=2, dim=1)
        n_in_y = nn.functional.normalize(in_y, p=2, dim=1)
        sims = (n_in_x @ n_in_y.t()).clamp(min=eps)
        return sims
    
    def return_loss_fn(self, x, t=0.5, eps=1e-8):
                
        ####################################################
        # Estimate cosine similarity intraA, intraB
        ####################################################
        domain_index = [p%2==0 for p in range(x.shape[0])]
        domain_index = torch.BoolTensor(domain_index).to(x.device)
        
        x_1 = x[domain_index]
        x_2 = x[~domain_index]
        
        #intraA & intraB
        x_a = self.get_similarities(x_1, x_1, eps)
        x_b = self.get_similarities(x_2, x_2, eps)
        
        ####################################################
        # Estimate cosine similarity interAB for SimCLR
        ####################################################
        
        x_ab = self.get_similarities(x, x, eps)

        ####################################################
        # QARe term
        ####################################################
        qare = self.qap_objective(x_a, x_b)
        
        ####################################################
        # LAP term / SimCLR
        ####################################################
        
        x_ab = torch.exp(x_ab / t)
        # Put positive pairs on the diagonal
        idx = torch.arange(x_ab.size()[0])
        idx[::2] += 1
        idx[1::2] -= 1
        x_ab = x_ab[idx]
        # subtract the similarity of 1 from the numerator
        x_ab = x_ab.diag() / (x_ab.sum(0) - torch.exp(torch.tensor(1 / t)))
        lap_term = -torch.log(x_ab.mean())
        
        ####################################################
        # combine
        ####################################################
        
        total_loss = self.alpha*lap_term + self.beta*qare

        return total_loss

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.net.train(True)
        self.head.train(True)
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
