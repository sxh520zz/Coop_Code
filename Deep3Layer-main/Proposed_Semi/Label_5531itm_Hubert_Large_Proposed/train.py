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
import torch.optim as optim
from torch.optim import AdamW


from utils import Get_data
from torch.autograd import Variable
from models import SpeechRecognitionModel
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold

from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
torch.backends.cudnn.enabled = False

with open('/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Semi/Hubert_Large_Proposed_5/Train_data.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=64, metavar='N')
parser.add_argument('--log_interval', type=int, default=10, metavar='N')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optim', type=str, default='AdamW')
parser.add_argument('--attention', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=4)
parser.add_argument('--utt_insize', type=int, default=1024)
args = parser.parse_args()

torch.manual_seed(args.seed)


def Train(epoch):
    train_loss = 0
    model.train()
    for batch_idx, (data_1,data_2, target) in enumerate(train_loader):
        if args.cuda:
            data_1,data_2, target = data_1.cuda(), data_2.cuda(), target.cuda()
        data_1,data_2, target = Variable(data_1), Variable(data_2),Variable(target)
        target = target.squeeze()
        utt_optim.zero_grad()
        data_1 = data_1.squeeze()
        data_2 = data_2.squeeze()
        utt_out = model(data_1,data_2)
        loss = torch.nn.CrossEntropyLoss()(utt_out, target.long())

        loss.backward()

        utt_optim.step()
        train_loss += loss

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval
            ))
            train_loss = 0

def Test():
    model.eval()
    label_pre = []
    label_true = []
    with torch.no_grad():
        for batch_idx, (data_1,data_2, target) in enumerate(test_loader):
            if args.cuda:
                data_1,data_2, target = data_1.cuda(), data_2.cuda(), target.cuda()
            data_1,data_2, target = Variable(data_1), Variable(data_2),Variable(target)
            target = target.squeeze()
            utt_optim.zero_grad()
            data_1 = data_1.squeeze()
            data_2 = data_2.squeeze()
            utt_out = model(data_1,data_2)
            output = torch.argmax(utt_out, dim=1)
            label_true.extend(target.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())
        accuracy_recall = recall_score(label_true, label_pre, average='macro')
        accuracy_f1 = metrics.f1_score(label_true, label_pre, average='macro')
        CM_test = confusion_matrix(label_true, label_pre)
        print(accuracy_recall)
        print(accuracy_f1)
        print(CM_test)
    return accuracy_f1, accuracy_recall, label_pre, label_true

Final_result = []
Fineal_f1 = []
result_label = []
kf = KFold(n_splits=5)
for index, (train, test) in enumerate(kf.split(data)):
    print(index)
    train_loader, test_loader, input_test_data_id, input_test_label_org = Get_data(data, train, test, args)
    model = SpeechRecognitionModel(args)
    #utt_net = Utterance_net(args.utt_insize, args.hidden_layer, args.out_class, args)
    if args.cuda:
        model = model.cuda()

    lr = args.lr
    utt_optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    utt_optim = optim.Adam(model.parameters(), lr=lr)
    f1 = 0
    recall = 0
    predict = copy.deepcopy(input_test_label_org)
    for epoch in range(1, args.epochs + 1):
        Train(epoch)
        accuracy_f1, accuracy_recall, pre_label, true_label = Test()
        print(len(pre_label))
        if (accuracy_recall > recall):
            num = 0
            for x in range(len(predict)):
                predict[x] = pre_label[num]
                num = num + 1
            result_label = predict
            recall = accuracy_recall
        print("Best Result Until Now:")
        print(recall)

    onegroup_result = []
    for i in range(len(input_test_data_id)):
        a = {}
        a['id'] = input_test_data_id[i]
        a['Predict_label'] = result_label[i]
        a['True_label'] = input_test_label_org[i]
        onegroup_result.append(a)
    Final_result.append(onegroup_result)
    Fineal_f1.append(recall)

true_label = []    
predict_label = []   
num = 0
for i in range(len(Final_result)):
    for j in range(len(Final_result[i])):
        num = num +1
        predict_label.append(Final_result[i][j]['Predict_label'])
        true_label.append(Final_result[i][j]['True_label'])
print(num)            
accuracy_recall = recall_score(true_label, predict_label, average='macro')
accuracy_f1 = metrics.f1_score(true_label, predict_label, average='macro')
CM_test = confusion_matrix(true_label,predict_label)

#-------------------------计算WA 和UA
predict_label = np.array(predict_label)
true_label = np.array(true_label)
wa = np.mean(predict_label.astype(int) == true_label.astype(int))

predict_label_onehot = np.eye(4)[predict_label.astype(int)]
true_label_onehot = np.eye(4)[true_label.astype(int)]
ua = np.mean(np.sum((predict_label_onehot == true_label_onehot)*true_label_onehot, axis =0 )/np.sum(true_label_onehot,axis =0))

print('UA={:.4f}, WA={:.4f}, F1={:.4f}' .format(ua,wa, accuracy_f1))
#print('WA={:.4f}'.format(wa))
#print(CM_test)
           
#print(accuracy_recall,accuracy_f1)
print(CM_test)    

'''
file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Semi/Hubert_Large_Proposed/Final_result.pickle', 'wb')
pickle.dump(Final_result,file)
file.close()
file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Semi/Hubert_Large_Proposed/Final_f1.pickle', 'wb')
pickle.dump(Fineal_f1,file)
file.close()
'''
