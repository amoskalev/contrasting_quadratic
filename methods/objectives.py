#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2022 Artem Moskalev
# Paper: "Contrasting quadratic assignments for set-based representation learning", A. Moskalev & I. Sosnovik & V. Fischer & A. Smeulders, ECCV 2022
# GitHub: https://github.com/amoskalev/contrasting_quadratic
#
# Implementation of the QARe objective / regularizer

import torch
import numpy as np
import torch.nn as nn

# QUADRATIC OBJECTIVES:
# COMPUTE EITHER EXACT LOWER-/UPPER-BOUND OR APPROX FROM https://www.sciencedirect.com/science/article/pii/0024379584901174
        
class QARe(nn.Module):
    def __init__(self):
        super().__init__()
    
    def max_scalar_prod(self, x, y):
        x_s = x.sort(descending=False).values
        y_s = y.sort(descending=False).values
        return x_s@y_s
    
    def min_scalar_prod(self, x, y):
        x_s = x.sort(descending=True).values
        y_s = y.sort(descending=False).values
        return x_s@y_s
    
    def forward(self, s1, s2, gt=None):
        
        B = s1.shape[-1]
        if gt is None:
            gt = torch.eye(B).to(s1.device)
        
        ###############################
        #0. Only positive similarities
        # assumes cos_sim as input
        ###############################
        s1 = 1 + s1
        s2 = 1 + s2
                
        ###############################
        # 3. compute QARe term
        ###############################
                
        #step1: eigenvalues term
        eig1 = torch.symeig(s1, eigenvectors=True).eigenvalues
        eig2 = torch.symeig(s2, eigenvectors=True).eigenvalues
        eig_term = self.max_scalar_prod(eig1, eig2)
        
        #step2: normalize
        norm_B = B*(B-1)
        eig_term_norm = eig_term/norm_B        
        
        qare_term = eig_term_norm
        
        return qare_term