import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import relu
import pandas as pd
import numpy as np

from transformers import AutoModel
class network0(nn.Module):
    def __init__(self, input_size = 768, hidden_size = 768, L = 3, bidirectional = True, num_LSTM_layers = 1, dropout = 0.2, embed = 'bert-base-cased', batch_size = 4):
        super().__init__()
        self.name = 'model0'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.L = L
        self.target = int(np.ceil(L/2)-1)
        self.D = 2 if bidirectional == True else 1
        self.batch_size = batch_size

        print("Loading " + embed + " Model...")
        self.tokenizer = embed
        self.embed = AutoModel.from_pretrained(embed)
        self.embed.train()
        print("Model Loaded.")

        self.LSTM = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers = num_LSTM_layers,
            bidirectional = bidirectional,
            batch_first=True)
        self.fc1 = nn.Linear(768, 192)
        self.fc2 = nn.Linear(192, 3)
        self.dropout = torch.nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def __call__(self, input):
        output = self.forward(input)
        return output
    
    def forward(self, x):
        '''x = torch.tensor_split(x, self.L, dim = 1)
        x = [self.embed(xi.squeeze())[1].reshape([self.batch_size,1,self.input_size]).squeeze() for xi in x]
        x = torch.cat(x,dim=1)'''
        x = x[:,1,:]
        x = self.embed(x)[1]
        
        #x = torch.cat([self.embed(xi.squeeze())[1].reshape([self.batch_size,1,self.input_size]) for xi in x],dim=1)
        #x = x.reshape(self.batch_size*self.L,x.shape[-1])
        #x = self.embed(x.squeeze())[1].reshape(self.batch_size, self.L, 768)
        #x = self.LSTM(x)[0][:,self.target,:]
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class network1(nn.Module):
    '''
    Network with 3 BERT embedding layers, taking S_T (target sentence), S_P (preceding sentence), S_F (following sentence).

    Embeds three sentences.
    Passes pooled outputs (CLS tokens) of each  sentenceas a sequence to a Bi-LSTM. 
    Takes middle hidden state of Bi-LSTM and passes through a two layer MLP.
    '''
    def __init__(self, input_size, hidden_size, L = 3, bidirectional = True, num_LSTM_layers = 1, dropout = 0.2, embed = 'bert-base-cased', batch_size = 64):
        super().__init__()
        self.name = 'model1'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.L = L
        self.target = int(np.ceil(L/2)-1)
        self.D = 2 if bidirectional == True else 1
        self.batch_size = batch_size

        print("Loading BERT Model")
        self.embed = AutoModel.from_pretrained(embed)
        self.embed.train()
        print("Model Loaded")

        self.LSTM = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers = num_LSTM_layers,
            bidirectional = bidirectional,
            batch_first=True)
        self.fc1 = nn.Linear(hidden_size*self.D, 192)
        self.fc2 = nn.Linear(192, 3)
        self.dropout = torch.nn.Dropout(dropout)
        

    def __call__(self, input):
        output = self.forward(input)
        return output
    
    def forward(self, x):
        x = torch.tensor_split(x, self.L, dim = 1)
        x = torch.cat([self.embed(xi.squeeze())[1].reshape([self.batch_size,1,self.input_size]) for xi in x],dim=1)
        #x = x.reshape(self.batch_size*self.L,x.shape[-1])
        #x = self.embed(x.squeeze())[1].reshape(self.batch_size, self.L, 768)
        x = self.LSTM(x)[0][:,self.target,:]
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class network2(nn.Module):
    '''
    Network with 3 BERT embedding layers, taking S_T (target sentence), S_P (preceding sentence), S_F (following sentence).

    Embeds three sentences using BERT transformers.
    Passes pooled outputs (CLS tokens) into two attention heads, computing attention between (S_T, S_P) and (S_T, S_F). 
    Concatenates outputs of attention heads and passes through a two layer MLP.
    '''
    def __init__(self, input_size = 768, hidden_size = 768, L = 3, bidirectional = True, num_att_heads = 1, dropout = 0.2, embed = 'bert-base-cased', batch_size = 4):
        super().__init__()
        self.name = 'model2'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.L = L
        self.target = int(np.ceil(L/2)-1)
        self.D = 2 if bidirectional == True else 1
        self.batch_size = batch_size

        print("Loading " + embed + " Model...")
        self.tokenizer = embed
        self.embed = AutoModel.from_pretrained(embed)
        self.embed.train()
        print("Model Loaded.")

        self.att1 = nn.MultiheadAttention(embed_dim=1, num_heads=num_att_heads,batch_first=True, dropout=dropout)
        self.att2 = nn.MultiheadAttention(embed_dim=1, num_heads=num_att_heads,batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size*self.D, 192)
        self.fc2 = nn.Linear(192, 3)
        self.dropout = torch.nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def __call__(self, input):
        output = self.forward(input)
        return output
    
    def forward(self, x):
    
        x = torch.tensor_split(x, self.L, dim = 1)
        x = [self.embed(x[i].squeeze())[1].reshape([self.batch_size,1,self.input_size]) for i in range(self.L)]
        x = torch.cat(x,dim=1)
        #x = torch.cat([self.embed(xi.squeeze())[1].reshape([self.batch_size,1,self.input_size]) for xi in x],dim=1)
        #x = x.reshape(self.batch_size*self.L,x.shape[-1])
        #x = self.embed(x.squeeze())[1].reshape(self.batch_size, self.L, 768)

        K = x[:,1,:].reshape((self.batch_size, self.input_size, 1))
        V = x[:,1,:].reshape((self.batch_size, self.input_size, 1))

        Q1 = x[:,0,:].reshape((self.batch_size, self.input_size, 1))
        Q2 = x[:,2,:].reshape((self.batch_size, self.input_size, 1))
        
        x1 = self.att1(Q1,K,V)
        x2 = self.att2(Q2,K,V)
        
        x = torch.cat([x1[0],x2[0]],dim=1).squeeze()
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class network3(nn.Module):
    '''
    Network with 3 BERT embedding layers, taking S_T (target sentence), S_P (preceding sentence), S_F (following sentence).

    Embeds three sentences using BERT transformers.
    Passes pooled outputs (CLS tokens) into two attention heads, computing attention between (S_T, S_P) and (S_T, S_F). 
    Concatenates outputs of attention heads and passes through a two layer MLP.
    '''
    def __init__(self, input_size = 768, hidden_size = 768, L = 3, bidirectional = True, num_att_heads = 1, dropout = 0.2, embed = 'bert-base-cased', batch_size = 4):
        super().__init__()
        self.name = 'model3'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.L = L
        self.target = int(np.ceil(L/2)-1)
        self.D = 2 if bidirectional == True else 1
        self.batch_size = batch_size

        print("Loading " + embed + " Model...")
        self.tokenizer = embed
        self.embed = AutoModel.from_pretrained(embed)
        self.embed.train()
        print("Model Loaded.")

        self.att1 = nn.MultiheadAttention(embed_dim=1, num_heads=num_att_heads,batch_first=True, dropout=dropout)
        self.att2 = nn.MultiheadAttention(embed_dim=1, num_heads=num_att_heads,batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size*3, 192)
        self.fc2 = nn.Linear(192, 3)
        self.dropout = torch.nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def __call__(self, input):
        output = self.forward(input)
        return output
    
    def forward(self, x):
    
        x = torch.tensor_split(x, self.L, dim = 1)
        x = [self.embed(x[i].squeeze())[1].reshape([self.batch_size,1,self.input_size]) for i in range(self.L)]
        x = torch.cat(x,dim=1)
        #x = torch.cat([self.embed(xi.squeeze())[1].reshape([self.batch_size,1,self.input_size]) for xi in x],dim=1)
        #x = x.reshape(self.batch_size*self.L,x.shape[-1])
        #x = self.embed(x.squeeze())[1].reshape(self.batch_size, self.L, 768)

        K = x[:,1,:].reshape((self.batch_size, self.input_size, 1))
        V = x[:,1,:].reshape((self.batch_size, self.input_size, 1))

        K1 = x[:,0,:].reshape((self.batch_size, self.input_size, 1))
        V1 = K1

        K2 = x[:,2,:].reshape((self.batch_size, self.input_size, 1))
        V2 = K2

        Q = x[:,0,:].reshape((self.batch_size, self.input_size, 1))
       
        
        x1 = self.att1(Q,K1,V1)
        x2 = self.att2(Q,K2,V2)
        
        x = torch.cat([x1[0],Q,x2[0]],dim=1).squeeze()
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

        
        