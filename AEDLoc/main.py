# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale
# import dnn
# import autoCode
import model
import Dataset
import os
import random
import yaml


# read param
base_file = open('./param/base_param.yaml','r',encoding='utf-8')
train_file = open('./param/train_param.yaml','r',encoding='utf-8')

base_param = yaml.load(base_file, Loader=yaml.FullLoader)
train_param = yaml.load(train_file, Loader=yaml.FullLoader)


train_path = base_param['train_path']
test_path = base_param['test_path']
input_channel = base_param['input_channel']
kernel_size = base_param['kernel_size']

epoch_1 = train_param['epoch_1']
epoch_2 = train_param['epoch_2']
epoch_3 = train_param['epoch_3']
fea_weights = train_param['fea_weights']
domain_weights = train_param['domain_weights']
pos_weights = train_param['pos_weights']
save = train_param['save']

batch_size = train_param['batch_size']
device = train_param['device']

train_dataset = Dataset.IndoorLocDataSet(train_path)
test_dataset = Dataset.IndoorLocDataSet(test_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
feature = model.FeatureNet(input_channel,kernel_size).to(device)
domain = model.DomainNet().to(device)
pos = model.PosNet().to(device)


def main():

    criterion = nn.CrossEntropyLoss(reduction="mean")
    criterion2 = nn.MSELoss()
    if fea_weights != []:
        feature.load_state_dict(torch.load(fea_weights[0]))
        pos.load_state_dict(torch.load(fea_weights[1]))
    model.train1(fea_net=feature, pred_net=pos, train_loader=train_loader, test_loader=train_loader, epochs=epoch_1,
                 criterion=criterion)
    if save:
        torch.save(feature.state_dict(), "fea1.pt")
        torch.save(pos.state_dict(), "pos1.pt")

    if domain_weights != "":
        domain.load_state_dict(torch.load(domain_weights))
    model.train2(fea_net=feature, domain_net=domain, train_loader=train_loader, test_loader=train_loader, epochs=epoch_2,
                 criterion=criterion)
    if save:
        torch.save(domain.state_dict(), "domain.pt")
    
    if pos_weights != []:
        feature.load_state_dict(torch.load(pos_weights[0]))
        pos.load_state_dict(torch.load(pos_weights[1]))
    model.train3(fea_net=feature, domain_net=domain, pred_net=pos, train_loader=train_loader, test_loader=test_loader,
                 criterion=criterion, epochs=epoch_3)
    if save:
        torch.save(feature.state_dict(), "fea2.pt")
        torch.save(pos.state_dict(), "pos2.pt")


def test():
    
    correct = 0
    total = 0
    pred = []
    real = []
    for data, label, _ in test_loader:
        data = data.to(device)
        label = label.to(device)
        fea = feature(data)
        pre = pos(fea)
        batch = label.size(0)
        total += batch
        for i in range(batch):
            if pre[i] == label[i]:
                correct += 1
            else:
                p = random.randint(0,10)
                if p < 1:
                    pred.append(pre[i].tolist())
                    real.append(label[i].tolist())
    print("acc:{:2f}%".format(100.0 * correct / total))
    print("pred: ", pred)
    print("real: ", real)


main()
test()
