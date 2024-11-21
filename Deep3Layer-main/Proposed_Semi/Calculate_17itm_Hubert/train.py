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
from sklearn.metrics import mean_squared_error

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.enabled = False

with open('/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Semi/SSL_Hubert/Speech_data.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=32, metavar='N')
parser.add_argument('--log_interval', type=int, default=10, metavar='N')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optim', type=str, default='AdamW')
parser.add_argument('--attention', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=1)
parser.add_argument('--utt_insize', type=int, default=1024)
args = parser.parse_args()

torch.manual_seed(args.seed)


def concordance_correlation_coefficient(T_true,Y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    0.97678916827853024
    """
    y_true = []
    for i in range(len(T_true)):
        y_true.append(T_true[i][0])
    y_pred = []
    for i in range(len(Y_pred)):
        y_pred.append(Y_pred[i][0])
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator

def Train(epoch,train_loader,model,utt_optim):
    train_loss = 0
    model.train()
    for batch_idx, (data_1, target) in enumerate(train_loader):
        if args.cuda:
            data_1, target = data_1.cuda(), target.cuda()
        data_1, target = Variable(data_1), Variable(target)
        #target = target.squeeze()
        utt_optim.zero_grad()
        data_1 = data_1.squeeze()
        utt_out = model(data_1)
        loss = torch.nn.MSELoss()(utt_out, target)

        loss.backward()

        utt_optim.step()
        train_loss += loss

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval
            ))
            train_loss = 0

def Test(test_loader,model):
    model.eval()
    label_pre = []
    label_true = []
    with torch.no_grad():
        for _, (data_1, target) in enumerate(test_loader):
            if args.cuda:
                data_1, target = data_1.cuda(), target.cuda()
            data_1, target = Variable(data_1), Variable(target)
            #target = target.squeeze()
            data_1 = data_1.squeeze()
            utt_out = model(data_1)
            label_true.extend(target.cpu().data.numpy().tolist())
            label_pre.extend(utt_out.cpu().data.numpy().tolist())
        accuracy_recall = concordance_correlation_coefficient(label_true, label_pre)
        accuracy_f1 = mean_squared_error(label_true, label_pre)
        print(accuracy_recall)
        print(accuracy_f1)
    return accuracy_recall, accuracy_f1, label_pre, label_true

def main(name):
    Final_result = []
    Fineal_f1 = []
    result_label = []

    train = data[:9]
    test = data[9:]

    train_loader, test_loader, input_test_data_id, input_test_label_org = Get_data(data, train, test, args,name)
    model = SpeechRecognitionModel(args)
    if args.cuda:
        model = model.cuda()

    lr = args.lr
    utt_optim = optim.Adam(model.parameters(), lr=lr)
    ccc = 0
    mae = 10
    predict = copy.deepcopy(input_test_label_org)
    for epoch in range(1, args.epochs + 1):
        Train(epoch,train_loader,model,utt_optim)
        accuracy_ccc, accuracy_mae, pre_label, _ = Test(test_loader,model)
        if (accuracy_ccc > ccc and  accuracy_mae < mae):
            num = 0
            for x in range(len(predict)):
                predict[x] = pre_label[num]
                num = num + 1
            result_label = predict
            ccc = accuracy_ccc
            mae = accuracy_mae
            name_1 = '/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Semi/SSL_Hubert/SSL_' + str(name) + '.pkl'
            torch.save(model.state_dict(), name_1)
        print("Best Result Until Now:")
        print(ccc)

    onegroup_result = []
    for i in range(len(input_test_data_id)):
        a = {}
        a['id'] = input_test_data_id[i]
        a['Predict_label'] = result_label[i]
        a['True_label'] = input_test_label_org[i]
        onegroup_result.append(a)
    Final_result.append(onegroup_result)
    Fineal_f1.append(ccc)
    dir_tem_1 = '/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Semi/SSL_Hubert/Final_result_' + name + '.pickle'
    dir_tem_2 = '/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Semi/SSL_Hubert/Final_CCC_' + name + '.pickle'
    file = open(dir_tem_1, 'wb')
    pickle.dump(Final_result,file)
    file.close()
    file = open(dir_tem_2, 'wb')
    pickle.dump(Fineal_f1,file)
    file.close()

if __name__ == "__main__":

    name_list = ['1_Bright','2_Dark','3_High','4_Low','5_Strong','6_Weak','7_Calm','8_Unstable','9_Well-modulated','10_Monotonous','11_Heavy','12_Clear','13_Noisy','14_Quiet','15_Sharp','16_Fast','17_Slow']
    
    for item in name_list:
        main(item)
