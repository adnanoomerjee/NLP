import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
from datetime import datetime
import random 

path = os.path.dirname(os.path.abspath(__file__))


class Get_Dataset(Dataset):
    def __init__(self, validate = False, train = True, pseudolabels = False, net0 = False):
        super().__init__()
        self.train = train
        self.pseudolabels = pseudolabels
        self.net0 = net0
        self.sentences, self.labels = self.load_data()
        #self.validation_split(validate)
        
    def __len__(self):
            return len(self.labels)
    
    def __getitem__(self, idx):
            labels = self.labels[idx]
            sentences = self.sentences[idx]
            sample = [sentences, labels]
            return sample
    
    def load_data(self):
        if self.train:
            sentences = np.load(str(path) + '/Data/train_sentences__13_03_2023__04_38_26.npy',allow_pickle=True)
            labels = np.load(str(path) + '/Data/train_labels__13_03_2023__04_38_26.npy',allow_pickle=True)
        else:
            sentences = np.load(str(path) + '/Data/test_sentences__13_03_2023__04_38_26.npy',allow_pickle=True).tolist()
            labels = np.load(str(path) + '/Data/test_labels__13_03_2023__04_38_26.npy',allow_pickle=True)

        labelled_indices = np.where(labels!=7)[0]
        labels = labels.tolist()

        if self.pseudolabels == False:
            sentences = [sentences[i] for i in labelled_indices]
            labels = [labels[i] for i in labelled_indices]
        if self.net0:
            labels = np.array(labels).squeeze()
            labels = labels - 1
            labels[np.where(labels == -1)] = 0

        
        return torch.tensor(sentences), torch.tensor(labels)

    def validation_split(self, validate):
        np.random.seed(0)
        ind = np.random.choice(len(self.labels), size = int(len(self.labels)*0.1))
        if validate:
            self.sentences = self.sentences[ind,:]
            self.labels = self.labels[ind]
        else:
            self.sentences = np.delete(self.sentences,ind, axis=0)
            self.labels = np.delete(self.labels,ind)
'''     
dataset = tokenizer()
trainset, testset = random_split(dataset, [0.8,0.2])
time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

trainset_name = str(path) + '/trainset__' + str(time)+'.pt'
testset_name = str(path) + '/testset__' + str(time)+'.pt'
torch.save(trainset, trainset_name)
torch.save(testset, testset_name)

class tokenizer(Dataset):
    def __init__(self, pseudolabels = False, num_context = 1, tokenizer = "bert-base-cased"):
        super().__init__()
        self.tokenizer = tokenizer
        self.autotokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.df = self.extract()
        self.sentences, self.labels = self.generate_word_dataset(df=self.df, num_context=num_context, pseudolabels = pseudolabels)

    def __len__(self):
            return len(self.labels)
    
    def __getitem__(self, idx):
            label = self.labels[idx]
            sentences = self.sentences[idx]
            sample = [sentences, label]
            return sample
    
    def extract(self):
        df_groundtruth = pd.read_csv(str(path)+'/../ClaimBuster_Datasets/datasets/groundtruth.csv')
        df_all_sentences = pd.read_csv(str(path)+'/../ClaimBuster_Datasets/datasets/all_sentences.csv')
        df_crowdsourced = pd.read_csv(str(path)+'/../ClaimBuster_Datasets/datasets/crowdsourced.csv')
        df_crwd_grnd = pd.concat([df_groundtruth,df_crowdsourced],axis=0)
        sentence_id_with_v = df_crwd_grnd['Sentence_id'].values
        df_without_label = df_all_sentences[~((df_all_sentences['Speaker'] == 'Information') | (df_all_sentences['Speaker_title'].isna()) | (df_all_sentences['Sentence_id'].isin(sentence_id_with_v))) ]
        df_without_label['Verdict'] = 6 #all non labels have 7 as verdict
        df_full = pd.concat([df_crwd_grnd,df_without_label],axis=0)
        df_full = df_full.sort_values('Sentence_id')
        df_full['Verdict'] = df_full['Verdict'].apply(lambda x: x+1)
        text = df_full["Text"].values.tolist()
        tokenized_text = self.autotokenizer(text, padding="longest", truncation=True)['input_ids']

        self.longest = len(tokenized_text[0])
    

        df_full['Text'] = tokenized_text
        df_full.reset_index(drop=True,inplace=True)
        
        return df_full
    
    def generate_word_dataset(self,df,num_context,pseudolabels):
        dataset = []
        label = []
        for index,row in df.iterrows():
            if index == 0 or index == len(df)-num_context:
                continue
            past = df.iloc[index-1]['Text']
            curr = row['Text']
            future = df.iloc[index+1]['Text']    
            if row['File_id'] != df.iloc[index-1]['File_id']:
                past = self.autotokenizer("",max_length = self.longest,padding="max_length", truncation=True)['input_ids']
            elif row['File_id'] != df.iloc[index+1]['File_id']:
                future = self.autotokenizer("",max_length = self.longest,padding="max_length", truncation=True)['input_ids']
            dataset.append([past,curr,future])
            label.append(row['Verdict'])
            
        label = np.array(label)

        labelled_indices = np.where(label!=7)[0]
        
        label = label.tolist()

        if pseudolabels == False:
            dataset = [dataset[i] for i in labelled_indices]
            label = [label[i] for i in labelled_indices]

        length = len(label)
        return torch.tensor(dataset), torch.tensor(label)

'''