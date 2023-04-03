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
    epochs = 15
    input_size = 768
    hidden_size = 768
    L = 3
    device = device()

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory = True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory = True, drop_last = True)


    def train(net, save=False):
        
        model = net

        ## loss and optimiser
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

        t0 = time.time()

        ## train
        for epoch in range(epochs):  # loop over the dataset multiple times
            
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                
                # print statistics
                running_loss += float(loss.item())
                if i%1000 == 0: #print update every 1000 batches
                    print('Epoch ' + str(epoch+1) + ', Batch ' + str((i+1)) + '/' + str(int(len(trainset)/batch_size)) + ', loss: ' + str(running_loss /(i+1)) + ', Time from start: ' +str(time.time()-t0))

            print('Epoch ' + str(epoch+1) + ', Batch ' + str((i+1)) + '/' + str(int(len(trainset)/batch_size)) + ', loss: ' + str(running_loss /(i+1)) + ', Time from start: ' +str(time.time()-t0))

            print('')

            y_true = []
            y_pred = []

            # run on test set
            with torch.no_grad():
                
                for i, data in enumerate(testloader, 0):
                    test_inputs, test_labels = data
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                    outputs = model(test_inputs)
                    _, predicted = torch.max(outputs, 1)

                    y_true.extend(test_labels.cpu().squeeze().tolist())
                    y_pred.extend(predicted.cpu().squeeze().tolist())

                    if i%100==0:
                        print('Epoch ' + str(epoch+1) + ', Loss: ' + str(running_loss /(len(testset)/batch_size)) + ', Running test: Batch ' + str(i+1) + '/' + str(int(len(testset)/batch_size)))
                print('Epoch ' + str(epoch+1) + ', Loss: ' + str(running_loss /(len(testset)/batch_size)) + ', Running test: Batch ' + str(i+1) + '/' + str(int(len(testset)/batch_size)))
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
                accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
                print(precision, recall, f1, accuracy)
                
                print('Epoch ' + str(epoch+1) + ', Loss: ' + str(running_loss /(len(testset)/batch_size)) + ', Test set accuracy: ' + str(accuracy.item()))
            
                
            print('')
        
            print('Training done.')

            model_name =str(path) + '/Model/' + net.name + '_checkpoint_epoch_' +str(epoch+1) +'.pt'

            if save:

                torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
                }, model_name)
                
                print('Model saved.')

    network1 = network1(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)
    train(network1)
    network2 = network2(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)
    train(network2)
    network0 = network0(input_size=input_size, hidden_size=hidden_size, L = L, batch_size = batch_size).to(device)
    train(network1)
    

    

