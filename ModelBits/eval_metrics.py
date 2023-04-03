import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from network import *
import time
import os
from get_dataset import Get_Dataset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import csv



def eval_run(save = False):

    path = os.path.dirname(os.path.abspath(__file__))


    def device():
        if torch.backends.mps.is_available():
            device = torch.device("mps") 
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device
    
    testset = Get_Dataset(train=False)
    testset0 = Get_Dataset(train=False, net0=True)
    
    batch_size = 4
    input_size = 768
    hidden_size = 768
    L = 3
    device = device()

    def evaluate(net, testset, save=False):
        
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True, drop_last = False)

        model = net
        t0 = time.time()
        
        ## eval metrics        
        y_true = []
        y_pred = []

        with torch.no_grad():
            print('\n')
            print('Running test batches...')
            for i, data in enumerate(testloader, 0):
                test_inputs, test_labels = data
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                outputs = model(test_inputs)
                _, predicted = torch.max(outputs, 1)

                y_true.extend(test_labels.cpu().squeeze().tolist())
                y_pred.extend(predicted.cpu().squeeze().tolist())
                if i%100==0:
                    print('Batch ' + str((i+1)) + '/' + str(int(len(testset)/batch_size)) + ', Time from start: ' +str(time.time()-t0))
            
            print('Batch ' + str((i+1)) + '/' + str(int(len(testset)/batch_size)) + ', Time from start: ' +str(time.time()-t0))
            print('Testing done.')
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
            confusion = confusion_matrix(y_true,y_pred, normalize='pred')
            #print(precision, recall, f1)
            weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        precision_dict = {'Precision': precision, 'Recall' : recall, 'F1': f1, 'Weighted Precision': weighted_precision, 'Weighted Recall' : weighted_recall, 'Weighted F1': weighted_f1}
        
        name= str(path) + '/Results/confusion_' +model.name+'_run.csv'
        if save:
            np.savetxt(name, confusion, delimiter=",")

        print('\n')
        print(precision_dict)
        print('\n')
        print('Confusion Matrix')
        print(confusion)
        print('\n')
        
        
        return precision_dict
    
    
    net0 = network0(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)
    net1 = network1(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)
    net2 = network2(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)


    networks = [net0, net1, net2]
    for i,network in enumerate(networks):
        print('\n')
        print(network.name)
        
        if i==0:
            test = testset0
        else:
            test = testset
        
        model_state = torch.load(str(path) + '/Model/model' +str(i)+'.pt',map_location = torch.device('cpu'))
        network.load_state_dict(model_state['model_state_dict'])
        network_df = evaluate(network,test)

        if save:
            with open(path + '/Results/' +network.name + ".csv", "w", newline="") as fp:
                # Create a writer object
                writer = csv.DictWriter(fp, fieldnames=network_df.keys())

                # Write the header row
                writer.writeheader()

                # Write the data rows
                writer.writerow(network_df)
            
        
if __name__ == "__main__":
    eval_run()





