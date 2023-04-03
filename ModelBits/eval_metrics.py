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
from sklearn.metrics import precision_recall_fscore_support
import csv

path = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

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

    
    #checkpoint = torch.load(str(path) + '/Model/model1_checkpoint_epoch_5.pt')
    #model.load_state_dict(checkpoint['model_state_dict'])

    def evaluate(net, testset):
        
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True, drop_last = False)

        model = net

        ## loss and optimiser
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, last_epoch =- 1)
        t0 = time.time()
        
        precisions = []
        recalls = []
        f1s = []
        accuracies = []
        weighted_precisions = []
        weighted_recalls = []
        weighted_f1s = []

        ## eval metrics        
        y_true = []
        y_pred = []

        with torch.no_grad():
            
            for i, data in enumerate(testloader, 0):
                test_inputs, test_labels = data
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                outputs = model(test_inputs)
                _, predicted = torch.max(outputs, 1)

                y_true.extend(test_labels.cpu().squeeze().tolist())
                y_pred.extend(predicted.cpu().squeeze().tolist())
                print('Batch ' + str((i+1)) + '/' + str(int(len(testset)/batch_size)) + ', Time from start: ' +str(time.time()-t0),end='\r')
                
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
            #print(precision, recall, f1)
            weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
            
            
        
        precision_dict = {'Precision': precision, 'Recall' : recall, 'F1': f1, 'Weighted Pecision': weighted_precision, 'Weighted Recall' : weighted_recall, 'Weighted F1': weighted_f1, 'Accuracy': accuracy}
        print(precision_dict)
        '''
        column_names = [precision]
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_precisions.append(weighted_precision)
        weighted_recalls.append(weighted_recall)
        weighted_f1s.append(weighted_f1)
        accuracies.append(accuracy)

        validation_scores = dict(network = model.__class__.__name__,
        precision = np.mean(precisions), 
        recall = np.mean(recalls), 
        f1_score = np.mean(f1s),
        weighted_precision = np.mean(weighted_precisions), 
        weighted_recall = np.mean(weighted_recalls), 
        weighted_f1_score = np.mean(weighted_f1s), 
        accuracy = np.mean(accuracies))
        
        validation_df = pd.DataFrame(validation_scores,index=[0])
        '''
        return precision_dict
    
    
    network0 = network0(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)
    network1 = network1(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)
    network2 = network2(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)
    network3 = network3(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)


    networks = [network0, network1, network2, network3]
    for i,network in enumerate(networks):
        a = 1
        if i!=a:
            continue
            test = testset0
        else:
            test = testset
        
        model_state = torch.load(str(path) + '/Model/model' +str(i)+'.pt',map_location = torch.device('cpu'))
        network.load_state_dict(model_state['model_state_dict'])
        network_df = evaluate(network,test)

        with open(path + '/Results/' +network.name + ".csv", "w", newline="") as fp:
            # Create a writer object
            writer = csv.DictWriter(fp, fieldnames=network_df.keys())

            # Write the header row
            writer.writeheader()

            # Write the data rows
            writer.writerow(network_df)
            
        








