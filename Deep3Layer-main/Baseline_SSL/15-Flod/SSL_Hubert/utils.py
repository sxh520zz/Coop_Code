import os
import time
import random
import argparse
import pickle
import copy
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.utils.rnn as rmm_utils
import torch.utils.data.dataset as Dataset
from sklearn import preprocessing

import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold


class subDataset(Dataset.Dataset):
    def __init__(self,Data_1,Label):
        self.Data_1 = Data_1
        self.Label = Label
    def __len__(self):
        return len(self.Data_1)
    def __getitem__(self, item):
        data_1 = torch.Tensor(self.Data_1[item])
        label = torch.Tensor(self.Label[item])
        return data_1,label
        
def STD(input_fea):
    data_1 = []
    for i in range(len(input_fea)):
        data_1.append(input_fea[i])
    a = []
    for i in range(len(data_1)):
        a.extend(data_1[i])
    scaler_1 = preprocessing.StandardScaler().fit(a)
    #print(scaler_1.mean_)
    #print(scaler_1.var_)
    for i in range(len(input_fea)):
        input_fea[i]   = scaler_1.transform(input_fea[i])
    return input_fea

def Feature(data,args):
    input_data_spec = []
    for i in range(len(data)):
        #print(len(data[i]['wav_encodings'][0][0]))
        input_data_spec.append(np.array(data[i]['wav_encodings'][0]))
        #input_data_spec.append(np.array(data[i]['adjective']).reshape(-1, 1))
    #input_data_spec = STD(input_data_spec)   

    name_list = ['1_Bright','2_Dark','3_High','4_Low','5_Strong','6_Weak','7_Calm','8_Unstable','9_Well-modulated','10_Monotonous','11_Heavy','12_Clear','13_Noisy','14_Quiet','15_Sharp','16_Fast','17_Slow']
    
    input_data_1 = []
    for i in range(len(data)):
        input_data_1.append(data[i]['1_Bright'])
    input_data_1 = np.array(input_data_1).reshape(-1, 1)
    input_data_2 = []
    for i in range(len(data)):
        input_data_2.append(data[i]['2_Dark'])
    input_data_2 = np.array(input_data_2).reshape(-1, 1)
    input_data_3 = []
    for i in range(len(data)):
        input_data_3.append(data[i]['3_High'])
    input_data_3 = np.array(input_data_3).reshape(-1, 1)
    input_data_4 = []
    for i in range(len(data)):
        input_data_4.append(data[i]['4_Low'])
    input_data_4 = np.array(input_data_4).reshape(-1, 1)
    input_data_5 = []
    for i in range(len(data)):
        input_data_5.append(data[i]['5_Strong'])
    input_data_5 = np.array(input_data_5).reshape(-1, 1)
    input_data_6 = []
    for i in range(len(data)):
        input_data_6.append(data[i]['6_Weak'])
    input_data_6 = np.array(input_data_6).reshape(-1, 1)
    input_data_7 = []
    for i in range(len(data)):
        input_data_7.append(data[i]['7_Calm'])
    input_data_7 = np.array(input_data_7).reshape(-1, 1)
    input_data_8 = []
    for i in range(len(data)):
        input_data_8.append(data[i]['8_Unstable'])
    input_data_8 = np.array(input_data_8).reshape(-1, 1)
    input_data_9 = []
    for i in range(len(data)):
        input_data_9.append(data[i]['9_Well-modulated'])
    input_data_9 = np.array(input_data_9).reshape(-1, 1)
    input_data_10 = []
    for i in range(len(data)):
        input_data_10.append(data[i]['10_Monotonous'])
    input_data_10 = np.array(input_data_10).reshape(-1, 1)
    input_data_11 = []
    for i in range(len(data)):
        input_data_11.append(data[i]['11_Heavy'])
    input_data_11 = np.array(input_data_11).reshape(-1, 1)
    input_data_12 = []
    for i in range(len(data)):
        input_data_12.append(data[i]['12_Clear'])
    input_data_12 = np.array(input_data_12).reshape(-1, 1)
    input_data_13 = []
    for i in range(len(data)):
        input_data_13.append(data[i]['13_Noisy'])
    input_data_13 = np.array(input_data_13).reshape(-1, 1)
    input_data_14 = []
    for i in range(len(data)):
        input_data_14.append(data[i]['14_Quiet'])
    input_data_14 = np.array(input_data_14).reshape(-1, 1)
    input_data_15 = []
    for i in range(len(data)):
        input_data_15.append(data[i]['15_Sharp'])
    input_data_15 = np.array(input_data_15).reshape(-1, 1)
    input_data_16 = []
    for i in range(len(data)):
        input_data_16.append(data[i]['16_Fast'])
    input_data_16 = np.array(input_data_16).reshape(-1, 1)
    input_data_17 = []
    for i in range(len(data)):
        input_data_17.append(data[i]['17_Slow'])
    input_data_17 = np.array(input_data_17).reshape(-1, 1)

    input_label = []
    for i in range(len(data)):
        input_label.append(data[i]['Cat']-1)
    input_data_id= []
    for i in range(len(data)):
        input_data_id.append(data[i]['Utterance'])
    input_label_org = []
    for i in range(len(data)):
        input_label_org.append(data[i]['Cat']-1)
    return input_data_spec,input_label,input_data_id,input_label_org

def Get_data(data,train,test,args):
    train_data = []
    test_data = []
    for i in range(len(train)):
        train_data.extend(data[train[i]])
    for i in range(len(test)):
        test_data.extend(data[test[i]])


    input_train_data_spec,input_train_label,_,_ = Feature(train_data,args)
    input_test_data_spec,input_test_label,input_test_data_id,input_test_label_org = Feature(test_data,args)


    #label = np.array(input_train_label, dype='int64').reshape(-1,1)
    #label_test = np.array(input_test_label, dype='int64').reshape(-1,1)
    label = np.array(input_train_label).reshape(-1, 1)
    label_test = np.array(input_test_label).reshape(-1,1)
    train_dataset = subDataset(input_train_data_spec,label)
    test_dataset = subDataset(input_test_data_spec,label_test)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,drop_last=False, shuffle=False)
    return train_loader,test_loader,input_test_data_id,input_test_label_org