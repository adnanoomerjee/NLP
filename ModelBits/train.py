import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from networks import *
import time
import os
from get_dataset import Get_Dataset
from sklearn.metrics import precision_recall_fscore_support

path = os.path.dirname(os.path.abspath(__file__))

def train(network, device, trainset, testset, batch_size=4, lr = 0.0005, num_workers = 0, epochs=20, test_on_epoch = True, save_on_epoch = False):

    '''
    if test_on_epoch:
        val_metrics = np.zeros((epochs,4))
    else:
        val_metrics = None
    '''

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory = True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory = True, drop_last = True)
    model = network.to(device)
    
    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5, last_epoch =- 1)
    t0 = time.time()
    ## train
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        network.train()
        
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.type(torch.LongTensor).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # print statistics
            running_loss += float(loss.item())
            print('Epoch ' + str(epoch+1) + ', Batch ' + str((i+1)) + '/' + str(int(len(trainset)/batch_size)) + ', Loss: ' + str(running_loss /(i+1)) + ', Time from start: ' +str(time.time()-t0), end = '\r')
        
        #scheduler.step()
        print('')

        if test_on_epoch:

            network.eval()
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
                    
                    print('Epoch ' + str(epoch+1) + ', Loss: ' + str(running_loss /(len(trainset)/batch_size)) + ', Running test: Batch ' + str(i+1) + '/' + str(int(len(testset)/batch_size)), end='\r')
 
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
                accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
                print(precision, recall, f1, accuracy)
                #val_metrics[epoch,:3] = np.array([precision, recall, f1])
                #val_metrics[epoch,3] = np.array([accuracy])

                print('Epoch ' + str(epoch+1) + ', Loss: ' + str(running_loss /(len(trainset)/batch_size)) + ', Test set accuracy: ' + str(accuracy.item()))
            
        print('')

        model_name =str(path) + '/Model/' + network.name + '_checkpoint_epoch_' +str(epoch+1) +'.pt'

        # save trained model
        #torch.save(model.state_dict(), model_name)
        if save_on_epoch:
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, model_name)
            
            print('Model saved.')

    print('Training done.')
    return val_metrics
 

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
    testset = Get_Dataset(train=True, validate=True)

    device = device()

    #checkpoint = torch.load(str(path) + '/Model/model1_checkpoint_epoch_5.pt')
    #model.load_state_dict(checkpoint['model_state_dict'])


    network = network1()
    train(network, device=device, trainset=trainset, testset=testset)