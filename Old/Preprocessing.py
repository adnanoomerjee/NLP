
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, TensorDataset
from transformers import AutoTokenizer
import os
from datetime import datetime

path = os.path.dirname(os.path.abspath(__file__))

class PreprocessData():
    def __init__(self, pseudolabels = False, num_context = 1, tokenizer = "bert-base-cased"):
        self.tokenizer = tokenizer
        self.autotokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.df = self.extract()
        self.trainset, self.testset = self.generate_word_dataset(df=self.df, num_context=num_context, pseudolabels = pseudolabels)
        
    def __len__(self):
        return len(self.trainset), len(self.testset)
    
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
 
        length = len(label)
        indices = np.arange(length)
        
        train_indices = np.random.choice(indices,int(length*0.8),replace=False)
        test_indices = np.delete(indices, train_indices)
        print(len(indices), len(train_indices), len(test_indices))
        dataset = np.array(dataset)
        label = np.array(label)

        train_sentences = dataset[train_indices]
        test_sentences = dataset[test_indices]
        train_labels = label[train_indices]
        test_labels = label[test_indices]

        print(train_sentences.shape)
        print(test_sentences.shape)
        print(train_labels.shape)
        print(test_labels.shape)
        return (train_sentences, train_labels), (test_sentences, test_labels)

dataset = PreprocessData()

train_sentences, train_labels = dataset.trainset
test_sentences, test_labels = dataset.testset

time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

train_sentences_name= str(path) + '/Data/train_sentences__' + str(time)
train_labels_name = str(path) + '/Data/train_labels__' + str(time)
test_sentences_name = str(path) + '/Data/test_sentences__' + str(time)
test_labels_name = str(path) + '/Data/test_labels__' + str(time)

np.save(train_sentences_name, train_sentences)
np.save(train_labels_name, train_labels)
np.save(test_sentences_name, test_sentences)
np.save(test_labels_name, test_labels)

'''dataset = PreprocessedData()
trainset, testset = random_split(dataset, [0.8,0.2])
trainset, testset = TensorDataset(trainset.numpy()), TensorDataset(testset.numpy())
print(type(trainset))
time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

trainset_name = str(path) + '/trainset__' + str(time)+'.pt'
testset_name = str(path) + '/testset__' + str(time)+'.pt'''
#torch.save(trainset, trainset_name)
#torch.save(testset, testset_name)

'''dataset = np.array(dataset)

        train_sentences = dataset[train_indices.astype(int)]
        test_sentences = dataset[test_indices]
        train_labels = label[train_indices]
        test_labels = label[test_indices]

        return train_sentences, train_labels, test_sentences, test_labels

train_sentences, train_labels, test_sentences, test_labels = PreprocessData()

time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")


train_sentences_name= str(path) + 'Data/train_sentences__' + str(time)
train_labels_name = str(path) + 'Data/test_sentences__' + str(time)
test_sentences_name = str(path) + 'Data/train_labels__' + str(time)
test_labels_name = str(path) + 'Data/test_labels__' + str(time)

np.save(train_sentences_name, train_sentences)
np.save(train_labels_name, test_sentences)
np.save(test_sentences_name, test_sentences_name)
np.save(test_labels_name, train_labels_name)

trainset = pd.DataFrame({'sentences':train_sentences,'labels':train_labels})
        testset = pd.DataFrame({'sentences':test_sentences,'labels':test_labels})
        
        return trainset, testset

trainset, testset = PreprocessData()

time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
trainset_name = str(path) + '/trainset__' + str(time)+'.csv'
testset_name = str(path) + '/testset__' + str(time)+'.csv'

trainset.to_csv(trainset_name)
trainset.to_csv(testset_name)

print(len(trainset, testset))

'''