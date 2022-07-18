#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2022 Artem Moskalev
# Paper: "Contrasting quadratic assignments for set-based representation learning", A. Moskalev & I. Sosnovik & V. Fischer & A. Smeulders, ECCV 2022
# GitHub: https://github.com/amoskalev/contrasting_quadratic
#
# auxilary utils

import torch
import torch.nn as nn

def constract_cost_matrix_eucl(left_features, right_features):
    diff = left_features.unsqueeze(1) - right_features.unsqueeze(0)
    return diff.norm(dim=-1)

def constract_cost_matrix_dot(left_features, right_features, normalize=False):
    
    if normalize:
        left_features = nn.functional.normalize(left_features, dim=-1, eps=1e-8)
        right_features = nn.functional.normalize(right_features, dim=-1, eps=1e-8)
    
    diff = left_features@right_features.T
    return diff