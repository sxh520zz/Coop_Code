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
from transformers import Wav2Vec2Model


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.enabled = False

with open('/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Deep3Layer/10-Flod/Deep3Layer_Hubert/Speech_data.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=15, metavar='N')
parser.add_argument('--log_interval', type=int, default=10, metavar='N')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optim', type=str, default='AdamW')
parser.add_argument('--attention', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=4)
parser.add_argument('--mid_class', type=float, default=1)
parser.add_argument('--utt_insize', type=int, default=1024)
args = parser.parse_args()

torch.manual_seed(args.seed)


def Train(epoch):
    train_loss = 0
    model.train()
    for batch_idx, (data_1, mid_1,mid_2,mid_3,mid_4,mid_5,mid_6,mid_7,mid_8,mid_9,mid_10,mid_11,mid_12,mid_13,mid_14,mid_15,mid_16,mid_17,target) in enumerate(train_loader):
        if args.cuda:
            data_1, mid_1,mid_2,mid_3,mid_4,mid_5,mid_6,mid_7,mid_8,mid_9,mid_10,mid_11,mid_12,mid_13,mid_14,mid_15,mid_16,mid_17,target = data_1.cuda(), mid_1.cuda(),mid_2.cuda(),mid_3.cuda(),mid_4.cuda(),mid_5.cuda(),mid_6.cuda(),mid_7.cuda(),mid_8.cuda(),mid_9.cuda(),mid_10.cuda(),mid_11.cuda(),mid_12.cuda(),mid_13.cuda(),mid_14.cuda(),mid_15.cuda(),mid_16.cuda(),mid_17.cuda(),target.cuda()
        data_1, mid_1,mid_2,mid_3,mid_4,mid_5,mid_6,mid_7,mid_8,mid_9,mid_10,mid_11,mid_12,mid_13,mid_14,mid_15,mid_16,mid_17, target = Variable(data_1), Variable(mid_1),Variable(mid_2),Variable(mid_3),Variable(mid_4),Variable(mid_5),Variable(mid_6),Variable(mid_7),Variable(mid_8),Variable(mid_9),Variable(mid_10),Variable(mid_11),Variable(mid_12),Variable(mid_13),Variable(mid_14),Variable(mid_15),Variable(mid_16),Variable(mid_17),Variable(target)
        
        target = target.squeeze()
        utt_optim.zero_grad()
        data_1 = data_1.squeeze()

        utt_out,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_10,label_11,label_12,label_13,label_14,label_15,label_16,label_17 = model(data_1)
        loss_1 = torch.nn.CrossEntropyLoss()(utt_out, target.long())
        loss_2_1 = torch.nn.MSELoss()(label_1, mid_1)
        loss_2_2 = torch.nn.MSELoss()(label_2, mid_2)
        loss_2_3 = torch.nn.MSELoss()(label_3, mid_3)
        loss_2_4 = torch.nn.MSELoss()(label_4, mid_4)
        loss_2_5 = torch.nn.MSELoss()(label_5, mid_5)
        loss_2_6 = torch.nn.MSELoss()(label_6, mid_6)
        loss_2_7 = torch.nn.MSELoss()(label_7, mid_7)
        loss_2_8 = torch.nn.MSELoss()(label_8, mid_8)
        loss_2_9 = torch.nn.MSELoss()(label_9, mid_9)
        loss_2_10 = torch.nn.MSELoss()(label_10, mid_10)
        loss_2_11 = torch.nn.MSELoss()(label_11, mid_11)
        loss_2_12 = torch.nn.MSELoss()(label_12, mid_12)
        loss_2_13 = torch.nn.MSELoss()(label_13, mid_13)
        loss_2_14 = torch.nn.MSELoss()(label_14, mid_14)
        loss_2_15 = torch.nn.MSELoss()(label_15, mid_15)
        loss_2_16 = torch.nn.MSELoss()(label_16, mid_16)
        loss_2_17 = torch.nn.MSELoss()(label_17, mid_17)

        loss = loss_1 + (loss_2_1 + loss_2_2 + loss_2_3 + loss_2_4 + loss_2_5 + loss_2_6 + loss_2_7 + loss_2_8 + loss_2_9 + loss_2_10 + loss_2_11 + loss_2_12 + loss_2_13 + loss_2_14 + loss_2_15 + loss_2_16 + loss_2_17)/17
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
    label_cont = []
    label_cont_1 = []
    label_cont_2 = []
    label_cont_3 = []
    label_cont_4 = []
    label_cont_5 = []
    label_cont_6 = []
    label_cont_7 = []
    label_cont_8 = []
    label_cont_9 = []
    label_cont_10 = []
    label_cont_11 = []
    label_cont_12 = []
    label_cont_13 = []
    label_cont_14 = []
    label_cont_15 = []
    label_cont_16 = []
    label_cont_17 = []   
    with torch.no_grad():
        for batch_idx, (data_1, mid_1,mid_2,mid_3,mid_4,mid_5,mid_6,mid_7,mid_8,mid_9,mid_10,mid_11,mid_12,mid_13,mid_14,mid_15,mid_16,mid_17,target) in enumerate(test_loader):
            if args.cuda:
                data_1, mid_1,mid_2,mid_3,mid_4,mid_5,mid_6,mid_7,mid_8,mid_9,mid_10,mid_11,mid_12,mid_13,mid_14,mid_15,mid_16,mid_17,target = data_1.cuda(), mid_1.cuda(),mid_2.cuda(),mid_3.cuda(),mid_4.cuda(),mid_5.cuda(),mid_6.cuda(),mid_7.cuda(),mid_8.cuda(),mid_9.cuda(),mid_10.cuda(),mid_11.cuda(),mid_12.cuda(),mid_13.cuda(),mid_14.cuda(),mid_15.cuda(),mid_16.cuda(),mid_17.cuda(),target.cuda()
            data_1, mid_1,mid_2,mid_3,mid_4,mid_5,mid_6,mid_7,mid_8,mid_9,mid_10,mid_11,mid_12,mid_13,mid_14,mid_15,mid_16,mid_17, target = Variable(data_1), Variable(mid_1),Variable(mid_2),Variable(mid_3),Variable(mid_4),Variable(mid_5),Variable(mid_6),Variable(mid_7),Variable(mid_8),Variable(mid_9),Variable(mid_10),Variable(mid_11),Variable(mid_12),Variable(mid_13),Variable(mid_14),Variable(mid_15),Variable(mid_16),Variable(mid_17),Variable(target)

            target = target.squeeze()
            data_1 = data_1.squeeze()
            utt_out,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_10,label_11,label_12,label_13,label_14,label_15,label_16,label_17 = model(data_1)
            output = torch.argmax(utt_out, dim=1)
            label_true.extend(target.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())
            label_cont_1.extend(label_1.cpu().data.numpy())
            label_cont_2.extend(label_2.cpu().data.numpy())
            label_cont_3.extend(label_3.cpu().data.numpy())
            label_cont_4.extend(label_4.cpu().data.numpy())
            label_cont_5.extend(label_5.cpu().data.numpy())
            label_cont_6.extend(label_6.cpu().data.numpy())
            label_cont_7.extend(label_7.cpu().data.numpy())
            label_cont_8.extend(label_8.cpu().data.numpy())
            label_cont_9.extend(label_9.cpu().data.numpy())
            label_cont_10.extend(label_10.cpu().data.numpy())
            label_cont_11.extend(label_11.cpu().data.numpy())
            label_cont_12.extend(label_12.cpu().data.numpy())
            label_cont_13.extend(label_13.cpu().data.numpy())
            label_cont_14.extend(label_14.cpu().data.numpy())
            label_cont_15.extend(label_15.cpu().data.numpy())
            label_cont_16.extend(label_16.cpu().data.numpy())
            label_cont_17.extend(label_17.cpu().data.numpy())
        label_cont.append(label_cont_1)
        label_cont.append(label_cont_2)
        label_cont.append(label_cont_3)
        label_cont.append(label_cont_4)
        label_cont.append(label_cont_5)
        label_cont.append(label_cont_6)
        label_cont.append(label_cont_7)
        label_cont.append(label_cont_8)
        label_cont.append(label_cont_9)
        label_cont.append(label_cont_10)
        label_cont.append(label_cont_11)
        label_cont.append(label_cont_12)
        label_cont.append(label_cont_13)
        label_cont.append(label_cont_14)
        label_cont.append(label_cont_15)
        label_cont.append(label_cont_16)
        label_cont.append(label_cont_17)
        accuracy_recall = recall_score(label_true, label_pre, average='macro')
        accuracy_f1 = metrics.f1_score(label_true, label_pre, average='macro')
        CM_test = confusion_matrix(label_true, label_pre)
        print(accuracy_recall)
        print(accuracy_f1)
        print(CM_test)
    return accuracy_f1, accuracy_recall, label_pre, label_true,label_cont

Final_result = []
Final_f1 = []
Final_cont = []
result_label = []
kf = KFold(n_splits=10)
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
    Cont = []
    predict = copy.deepcopy(input_test_label_org)
    for epoch in range(1, args.epochs + 1):
        Train(epoch)
        accuracy_f1, accuracy_recall, pre_label, true_label,label_cont = Test()
        if (accuracy_recall > recall and accuracy_f1 > f1):
            num = 0
            for x in range(len(predict)):
                predict[x] = pre_label[num]
                num = num + 1
            result_label = predict
            recall = accuracy_recall
            f1 = accuracy_f1
            Cont = label_cont
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
    Final_f1.append(recall)
    Final_cont.append(Cont)
file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Deep3Layer/10-Flod/Deep3Layer_Hubert/Final_result.pickle', 'wb')
pickle.dump(Final_result,file)
file.close()
file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Deep3Layer/10-Flod/Deep3Layer_Hubert/Final_f1.pickle', 'wb')
pickle.dump(Final_f1,file)
file.close()
file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Deep3Layer/10-Flod/Deep3Layer_Hubert/Final_cont.pickle', 'wb')
pickle.dump(Final_cont,file)
file.close()