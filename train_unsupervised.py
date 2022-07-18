#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2022 Artem Moskalev
# Copyright (c) 2020 Massimiliano Patacchiola
# Paper: "Contrasting quadratic assignments for set-based representation learning", A. Moskalev & I. Sosnovik & V. Fischer & A. Smeulders, ECCV 2022
# GitHub: https://github.com/amoskalev/contrasting_quadratic

import os
import argparse

parser = argparse.ArgumentParser(description="Contrastive training script")
parser.add_argument("--seed", default=-1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
parser.add_argument("--epoch_start", default=0, type=int, help="Epoch to start learning from, used when resuming")
parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs")
parser.add_argument("--dataset", default="cifar10", help="Dataset: cifar10|100, supercifar100, tiny, slim, stl10")
parser.add_argument("--backbone", default="conv4", help="Backbone: conv4, resnet|8|32|34|56")
parser.add_argument("--method", default="simclr", help="simclr, simclr_qare, sparceclr, sparceclr_qare")

# backbone & head args
parser.add_argument("--inner_fs", default=64, type=int, help="Output feature dimension of the backbone network")
parser.add_argument("--outter_fs", default=64, type=int, help="Output feature dimension of the head")
parser.add_argument("--eval_data_size", default=128, type=int, help="Batch size to train network for linear evaluation")

# qare args
parser.add_argument("--alpha", default=1, type=float, help="weighting for linear term")
parser.add_argument("--beta", default=1, type=float, help="weighting for qare term")

parser.add_argument("--data_size", default=128, type=int, help="Size of the mini-batch")
parser.add_argument("--K", default=32, type=int, help="Total number of augmentations (K), sed only in RelationNet")
parser.add_argument("--aggregation", default="cat", help="Aggregation function used in RelationNet: sum, mean, max, cat")
parser.add_argument("--id", default="", help="Additional string appended when saving the checkpoints")
parser.add_argument("--checkpoint", default="", help="location of a checkpoint file, used to resume training")
parser.add_argument("--num_workers", default=8, type=int, help="Number of torchvision workers used to load data (default: 8)")
parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")
args = parser.parse_args()

args_id = "{}_{}".format(args.alpha, args.beta)
header = str(args.method)+ "_" + str(args_id) + "_" + str(args.dataset) + "_" + str(args.backbone) + "_seed_" + str(args.seed)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import random

if(args.seed>=0):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    print("[INFO] Setting SEED: " + str(args.seed))   
else:
    print("[INFO] Setting SEED: None")

if(torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")
                                          
if(args.backbone=="conv4"):
    from backbones.conv4 import Conv4
    feature_extractor = Conv4(flatten=True)
elif(args.backbone=="resnet8"):
    from backbones.resnet_small import ResNet, BasicBlock
    feature_extractor = ResNet(BasicBlock, [1, 1, 1], channels=[16, 32, 64], flatten=True)
elif(args.backbone=="resnet32"):
    from backbones.resnet_small import ResNet, BasicBlock
    feature_extractor = ResNet(BasicBlock, [5, 5, 5], channels=[16, 32, 64], flatten=True)
elif(args.backbone=="resnet56"):
    from backbones.resnet_small import ResNet, BasicBlock
    feature_extractor = ResNet(BasicBlock, [9, 9, 9], channels=[16, 32, 64], flatten=True)
elif(args.backbone=="resnet34"):
    from backbones.resnet_large import ResNet, BasicBlock
    feature_extractor = ResNet(BasicBlock, layers=[3, 4, 6, 3],zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None)          
else:
    raise RuntimeError("[ERROR] the backbone " + str(args.backbone) +  " is not supported.")

tot_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
print("[INFO]", str(str(args.backbone)), "loaded in memory.")
print("[INFO] Inner fs: {}".format(str(args.inner_fs)))
print("[INFO] Outter fs: {}".format(str(args.outter_fs)))
print("[INFO] alpha:{}, beta:{}".format(args.alpha, args.beta))
print("[INFO] Feature extractor TOT trainable params: {}".format(str(tot_params)))
print("[INFO] Found {} GPU(s) available.".format(str(torch.cuda.device_count())))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device type: {}".format(str(device)))

# The datamanager is a separate class that returns
# appropriate data loaders and information based
# on the method and dataset selected.
from datamanager import DataManager
manager = DataManager(args.seed)
num_classes = manager.get_num_classes(args.dataset)

if(args.method=="standard"):

    from methods.standard import StandardModel
    model = StandardModel(feature_extractor, num_classes, tot_epochs=args.epochs)
    if(args.dataset=="stl10"):
        train_transform = manager.get_train_transforms("finetune", args.dataset)
    else:
        train_transform = manager.get_train_transforms("standard", args.dataset)
    test_loader = manager.get_test_loader(args.dataset, data_size=args.data_size, num_workers=args.num_workers)
    train_loader, _ = manager.get_train_loader(dataset=args.dataset, 
                                            data_type="single",
                                            data_size=args.data_size, 
                                            train_transform=train_transform, 
                                            repeat_augmentations=None,
                                            num_workers=args.num_workers,
                                            drop_last=False)
elif(args.method=="rotationnet"):
    from methods.rotationnet import Model
    model = Model(feature_extractor)
    train_transform = manager.get_train_transforms(args.method, args.dataset)
    if(args.dataset=="stl10"): data_type="unsupervised"
    else: data_type="single"
    train_loader, _ = manager.get_train_loader(dataset=args.dataset, 
                                            data_type=data_type, 
                                            data_size=args.data_size, 
                                            train_transform=train_transform,
                                            repeat_augmentations=None,
                                            num_workers=args.num_workers,
                                            drop_last=False)

elif(args.method=="randomweights"):

    if not os.path.exists("./checkpoint/"+str(args.method)+"/"+str(args.dataset)): 
        os.makedirs("./checkpoint/"+str(args.method)+"/"+str(args.dataset))
    feature_extractor_state_dict = feature_extractor.state_dict()
    checkpoint_path = "./checkpoint/"+str(args.method)+"/"+str(args.dataset)+"/"+header+".tar"
    print("Saving in:", checkpoint_path)
    torch.save({"backbone": feature_extractor_state_dict}, checkpoint_path)
    import sys
    sys.exit() 

elif(args.method=="deepinfomax"):

    from methods.deepinfomax import DIM
    model = DIM(feature_extractor, alpha=0.0, beta=1.0, gamma=0.1)
    train_transform = manager.get_train_transforms(args.method, args.dataset)
    if(args.dataset=="stl10"): data_type="unsupervised"
    else: data_type="single"
    train_loader, _ = manager.get_train_loader(dataset=args.dataset, 
                                            data_type=data_type, 
                                            data_size=args.data_size, 
                                            train_transform=train_transform,
                                            repeat_augmentations=None,
                                            num_workers=args.num_workers,
                                            drop_last=True)

elif(args.method=="simclr"):    
    from methods.simclr import Model
    model = Model(feature_extractor)
    train_transform = manager.get_train_transforms(args.method, args.dataset)
    train_loader, _ = manager.get_train_loader(dataset=args.dataset,
                                            data_type="multi",
                                            data_size=args.data_size,
                                            train_transform=train_transform,
                                            repeat_augmentations=2,
                                            num_workers=args.num_workers,
                                            drop_last=False)

elif(args.method=="simclr_qare"):    
    from methods.simclr_qare import Model
    model = Model(feature_extractor, out_dim=args.outter_fs, alpha=args.alpha, beta=args.beta)
    train_transform = manager.get_train_transforms('simclr', args.dataset)
    train_loader, _ = manager.get_train_loader(dataset=args.dataset,
                                               data_type="multi",
                                               data_size=args.data_size,
                                               train_transform=train_transform,
                                               repeat_augmentations=2,
                                               num_workers=args.num_workers,
                                               drop_last=False)
elif(args.method=="sparseclr"):    
    from methods.sparseclr import Model
    model = Model(feature_extractor, out_dim=args.outter_fs)
    train_transform = manager.get_train_transforms('simclr', args.dataset)
    train_loader, _ = manager.get_train_loader(dataset=args.dataset,
                                               data_type="multi",
                                               data_size=args.data_size,
                                               train_transform=train_transform,
                                               repeat_augmentations=2,
                                               num_workers=args.num_workers,
                                               drop_last=False)
elif(args.method=="sparseclr_qare"):    
    from methods.sparseclr_qare import Model
    model = Model(feature_extractor, out_dim=args.outter_fs, alpha=args.alpha, beta=args.beta)
    train_transform = manager.get_train_transforms('simclr', args.dataset)
    train_loader, _ = manager.get_train_loader(dataset=args.dataset,
                                               data_type="multi",
                                               data_size=args.data_size,
                                               train_transform=train_transform,
                                               repeat_augmentations=2,
                                               num_workers=args.num_workers,
                                               drop_last=False)

elif(args.method=="deepcluster"):    
    from methods.deepcluster import Model
    train_transform = manager.get_train_transforms(args.method, args.dataset)
    model = Model(feature_extractor, batch_size=args.data_size, num_clusters=num_classes*10, train_transform=train_transform)
    if(args.dataset=="stl10"): data_type="unsupervised"
    else: data_type="single"
    _, train_set = manager.get_train_loader(dataset=args.dataset,
                                         data_type=data_type,
                                         data_size=args.data_size,
                                         train_transform=train_transform,
                                         repeat_augmentations=None,
                                         num_workers=args.num_workers,
                                         drop_last=False)
    # Note: for DeepCluster we take the train-set but here for convinience
    # we rename it train_loader to avoid overhead in the training loop.
    train_loader = train_set
else:
    raise RuntimeError("[ERROR] the model " + str(args.method) +  " is not supported.")


model.to(device)

# NOTE: the checkpoint must be loaded AFTER 
# the model has been allocated into the device.
if(args.checkpoint!=""):
    print("Loading checkpoint: " + str(args.checkpoint))
    model.load(args.checkpoint)
    print("Loading checkpoint: Done!")

def main():

    ########################################################
    # TRAIN UNSUPERVISED
    ########################################################

    #DATASET ARCH BATCH_SIZE INNER_FS OUTTER_FS ALPHA BETA
    make_name = "./checkpoint/"+str(args.method)+'_D:{}_Ar:{}_BS:{}_IFS:{}_OFS:{}_A:{}_B:{}'.format(args.dataset, 
                                                                                                    args.backbone, 
                                                                                                    args.data_size, 
                                                                                                    args.inner_fs, 
                                                                                                    args.outter_fs, 
                                                                                                    args.alpha, 
                                                                                                    args.beta)                                         
    if not os.path.exists(make_name): 
        os.makedirs(make_name)
    log_file = make_name+"/log_"+header+".cvs"
    with open(log_file, "w") as myfile: myfile.write("epoch,loss,score" + "\n") # create a new log file (it destroys the previous one)
    for epoch in range(args.epoch_start, args.epochs):
        loss_train, accuracy_train = model.train(epoch, train_loader) #<-- Each model must have a "train" method
        with open(log_file, "a") as myfile:
                myfile.write(str(epoch)+","+str(loss_train)+","+str(accuracy_train)+"\n")        
        if(epoch in [int(args.epochs*0.25)-1, int(args.epochs*0.5)-1, int(args.epochs*0.75)-1, args.epochs-1]):
            checkpoint_path = make_name+"/"+header+"_epoch_"+ str(epoch+1)+".tar"
            print("[INFO] Saving in:", checkpoint_path)            
            model.save(checkpoint_path)
    if(args.method=="standard"):
        #For the standard supervised method, it estimates now the test accuracy
        loss_test, accuracy_test = model.test(test_loader)
        print("Test loss: " + str(loss_test) )
        print("Test accuracy: " + str(accuracy_test) + "%")
if __name__== "__main__": main()
