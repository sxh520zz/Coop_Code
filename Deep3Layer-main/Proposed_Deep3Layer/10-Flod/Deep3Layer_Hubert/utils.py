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
    def __init__(self,Data_1,Mid_1,Mid_2,Mid_3,Mid_4,Mid_5,Mid_6,Mid_7,Mid_8,Mid_9,Mid_10,Mid_11,Mid_12,Mid_13,Mid_14,Mid_15,Mid_16,Mid_17,Label):
        self.Data_1 = Data_1
        self.Mid_1 = Mid_1
        self.Mid_2 = Mid_2
        self.Mid_3 = Mid_3
        self.Mid_4 = Mid_4
        self.Mid_5 = Mid_5
        self.Mid_6 = Mid_6
        self.Mid_7 = Mid_7
        self.Mid_8 = Mid_8
        self.Mid_9 = Mid_9
        self.Mid_10 = Mid_10
        self.Mid_11 = Mid_11
        self.Mid_12 = Mid_12
        self.Mid_13 = Mid_13
        self.Mid_14 = Mid_14
        self.Mid_15 = Mid_15
        self.Mid_16 = Mid_16
        self.Mid_17 = Mid_17
        self.Label = Label
    def __len__(self):
        return len(self.Data_1)
    def __getitem__(self, item):
        data_1 = torch.Tensor(self.Data_1[item])
        mid_1 = torch.Tensor(self.Mid_1[item])
        mid_2 = torch.Tensor(self.Mid_2[item])
        mid_3 = torch.Tensor(self.Mid_3[item])
        mid_4 = torch.Tensor(self.Mid_4[item])
        mid_5 = torch.Tensor(self.Mid_5[item])
        mid_6 = torch.Tensor(self.Mid_6[item])
        mid_7 = torch.Tensor(self.Mid_7[item])
        mid_8 = torch.Tensor(self.Mid_8[item])
        mid_9 = torch.Tensor(self.Mid_9[item])
        mid_10 = torch.Tensor(self.Mid_10[item])
        mid_11 = torch.Tensor(self.Mid_11[item])
        mid_12 = torch.Tensor(self.Mid_12[item])
        mid_13 = torch.Tensor(self.Mid_13[item])
        mid_14 = torch.Tensor(self.Mid_14[item])
        mid_15 = torch.Tensor(self.Mid_15[item])
        mid_16 = torch.Tensor(self.Mid_16[item])
        mid_17 = torch.Tensor(self.Mid_17[item])
        label = torch.Tensor(self.Label[item])
        return data_1,mid_1,mid_2,mid_3,mid_4,mid_5,mid_6,mid_7,mid_8,mid_9,mid_10,mid_11,mid_12,mid_13,mid_14,mid_15,mid_16,mid_17,label
        
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
    return input_data_spec,input_data_1,input_data_2,input_data_3,input_data_4,input_data_5,input_data_6,input_data_7,input_data_8,input_data_9,input_data_10,input_data_11,input_data_12,input_data_13,input_data_14,input_data_15,input_data_16,input_data_17,input_label,input_data_id,input_label_org

def Get_data(data,train,test,args):
    train_data = []
    test_data = []
    for i in range(len(train)):
        train_data.extend(data[train[i]])
    for i in range(len(test)):
        test_data.extend(data[test[i]])


    input_train_data_spec,input_data_1,input_data_2,input_data_3,input_data_4,input_data_5,input_data_6,input_data_7,input_data_8,input_data_9,input_data_10,input_data_11,input_data_12,input_data_13,input_data_14,input_data_15,input_data_16,input_data_17,input_train_label,_,_ = Feature(train_data,args)
    input_test_data_spec,input_data_1_test,input_data_2_test,input_data_3_test,input_data_4_test,input_data_5_test,input_data_6_test,input_data_7_test,input_data_8_test,input_data_9_test,input_data_10_test,input_data_11_test,input_data_12_test,input_data_13_test,input_data_14_test,input_data_15_test,input_data_16_test,input_data_17_test, input_test_label,input_test_data_id,input_test_label_org = Feature(test_data,args)


    #label = np.array(input_train_label, dype='int64').reshape(-1,1)
    #label_test = np.array(input_test_label, dype='int64').reshape(-1,1)
    label = np.array(input_train_label).reshape(-1, 1)
    label_test = np.array(input_test_label).reshape(-1,1)
    train_dataset = subDataset(input_train_data_spec, input_data_1,input_data_2,input_data_3,input_data_4,input_data_5,input_data_6,input_data_7,input_data_8,input_data_9,input_data_10,input_data_11,input_data_12,input_data_13,input_data_14,input_data_15,input_data_16,input_data_17,label)
    test_dataset = subDataset(input_test_data_spec, input_data_1_test,input_data_2_test,input_data_3_test,input_data_4_test,input_data_5_test,input_data_6_test,input_data_7_test,input_data_8_test,input_data_9_test,input_data_10_test,input_data_11_test,input_data_12_test,input_data_13_test,input_data_14_test,input_data_15_test,input_data_16_test,input_data_17_test,label_test)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,drop_last=False, shuffle=False)
    return train_loader,test_loader,input_test_data_id,input_test_label_org