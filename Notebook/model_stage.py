
from transformers import AutoModel
from torch.utils.data import DataLoader
import numpy as np
import torch
import time
import os
path = os.path.dirname(os.path.abspath(__file__))
'''trainset = PreprocessedData()
print(trainset)
batch_size = 64
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True)

sentences, labels = next(iter(trainloader))
split = torch.tensor_split(sentences,3,dim=1)
t1 = time.time()
model = AutoModel.from_pretrained('bert-base-cased')
t2 = time.time()
print(t2-t1)
print([model(s.squeeze())[0] for s in split])'''
labels = np.load(str(path) + '/Data/train_labels__13_03_2023__03_56_29.npy').tolist()
print(labels.shape)