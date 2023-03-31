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
    
    trainset = Get_Dataset(train=True)
    testset = Get_Dataset(train=False, validate=True)
    
    batch_size = 4
    epochs = 5
    input_size = 768
    hidden_size = 768
    L = 3
    device = device()

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory = True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory = True, drop_last = True)


    #checkpoint = torch.load(str(path) + '/Model/model1_checkpoint_epoch_5.pt')
    #model.load_state_dict(checkpoint['model_state_dict'])

    def evaluate(net):
        
        model = net

        ## loss and optimiser
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, last_epoch =- 1)
        t0 = time.time()
        
        precisions = []
        recalls = []
        f1s = []
        accuracies = []
        weighted_precisions = []
        weighted_recalls = []
        weighted_f1s = []

        ## eval metrics per epoch
        for epoch in range(epochs):  # loop over the dataset multiple times
            
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
 
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
                weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
                print('Epoch ' + str(epoch+1) + '\n, Pecision: {precision}, Recall : {recall}, F1: {f1}\n Weighted Pecision: {weighted_precision}, Weighted Recall : {weighted_recall}, Weighted F1: {weighted_f1}')

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

        validation_df = pd.DataFrame.from_dict(validation_scores)

        return validation_df

    network1 = network1(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)
    network1_df = evaluate(network1)

    network2 = network2(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)
    network2_df = evaluate(network2)

    scores = network2.append(network2)

    print("Evaluation metrics:")
    print(scores)




