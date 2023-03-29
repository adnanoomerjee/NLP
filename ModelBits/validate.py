import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from get_dataset import Get_Dataset
from networks import network1, network2
from train import train
import pickle
import os

path = os.path.dirname(os.path.abspath(__file__))

def device():
    if torch.backends.mps.is_available():
        device = torch.device("mps") 
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

device = device()

def cross_validate(network, lr, epochs = 10, num_workers = 0, k=4):
    X = Get_Dataset(train=True)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42) # Change random states
    
    crossvalrun_metrics = np.zeros((epochs, 4, k))

    for i, (train_index, val_index) in enumerate(kf.split(X)):
    
        print('Fold ' + str(i+1) + '/' +str(k))
        print('')
        trainset = Get_Dataset(train=True, fold_indices=train_index)
        valset = Get_Dataset(train=True, fold_indices=val_index)

        train_run = train(device=device, network=network, trainset=trainset, testset = valset, lr = lr, num_workers = num_workers, epochs = epochs)

        crossvalrun_metrics[:,:,i] = train_run
        
    crossvalrun_out = crossvalrun_metrics.mean(axis=2)
    
    return crossvalrun_out.squeeze()

if __name__ == "__main__":

    learning_rates = [0.001, 0.0005, 0.0003, 0.0001]
    networks = [network1(), network2()]

    num_workers = 0
    epochs = 5

    lr_metrics_1 = {}
    lr_metrics_2 = {}

    for n, network in enumerate(networks):
        for lr in learning_rates:
            print('')
            print('Cross Validation, Network ' + str(n+1) + ', Learning Rate = ' + str(lr))
            print('\n')

            crossvalmetrics_lr = cross_validate(network, lr = lr,num_workers=num_workers, epochs=epochs)

            if n==0:
                lr_metrics_1[lr] = crossvalmetrics_lr
            else:
                lr_metrics_2[lr] = crossvalmetrics_lr

    with open(path+'/cross_validation_outputs_net_1.pkl', 'wb') as f:
        pickle.dump(lr_metrics_1, f)

    with open(path+'/cross_validation_outputs_net_2.pkl', 'wb') as f:
        pickle.dump(lr_metrics_2, f)

    print(lr_metrics_1)
    print(lr_metrics_2)