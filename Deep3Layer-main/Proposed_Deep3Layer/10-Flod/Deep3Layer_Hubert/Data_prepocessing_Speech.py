import pandas as pd
import numpy as np
import python_speech_features as ps
import soundfile as sf
import os
from sklearn import preprocessing
import pickle
import torch
import torchaudio
import subprocess
from transformers import AutoProcessor, HubertModel


rootdir = os.path.dirname(os.path.abspath('..'))
print(rootdir)
data_file = '/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Deep3Layer_XShi_XLi/sound538/'
data_1_file = '/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Deep3Layer_XShi_XLi/Down_sound/'
label_file = '/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Deep3Layer_XShi_XLi/BCF_Origin_Semantics_new.xlsx'

from transformers import AutoProcessor, HubertModel
processor = AutoProcessor.from_pretrained("/mnt/data1/liyongwei/SSL_Models/facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("/mnt/data1/liyongwei/SSL_Models/facebook/hubert-large-ls960-ft")   # 用于提取通用特征，768维

def change_sample_rate(input_file, target_sample_rate, output_file):
    """
    改变音频信号的采样率为目标采样率。

    参数：
        input_file (str): 输入音频文件的路径。
        target_sample_rate (int): 目标采样率（单位：Hz）。
        output_file (str): 输出音频文件的路径。
    """
    # 使用 ffmpeg 命令进行采样率改变，并保存为输出文件
    subprocess.run(["ffmpeg", "-i", input_file, "-ar", str(target_sample_rate), output_file])

def downsample_folder(input_folder, output_folder, target_sample_rate):
    """
    降采样整个文件夹中的音频文件。

    参数：
        input_folder (str): 输入文件夹路径，包含需要降采样的音频文件。
        output_folder (str): 输出文件夹路径，用于保存降采样后的音频文件。
        target_sample_rate (int): 目标采样率（单位：Hz）。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的音频文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            change_sample_rate(input_file, target_sample_rate, output_file)

def process_wav_file(wav_file, time):
    waveform, sample_rate = torchaudio.load(wav_file)
    target_length = time * sample_rate
    # 将WAV文件裁剪为目标长度
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    else:
        # 如果WAV文件长度小于目标长度，则使用填充进行扩展
        padding_length = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding_length))

    return waveform, sample_rate

def STD(input_fea,name):
    data_1 = []
    for i in range(len(input_fea)):
        data_1.append(input_fea[i][name])
    a = []
    for i in range(len(data_1)):
        a.extend(data_1[i])
    scaler_1 = preprocessing.MinMaxScaler().fit(a)
    #print(scaler_1.mean_)
    #print(scaler_1.var_)
    for i in range(len(input_fea)):
        input_fea[i][name] = scaler_1.transform(input_fea[i][name])
    return input_fea

def read_excel_file(file_path):
    """
    读取 .xlsx 文件并返回其中的信息。

    参数：
        file_path (str): .xlsx 文件的路径。

    返回：
        dict: 包含文件中信息的字典。
             字典的键为行号（从0开始），值为对应行的数据（作为一个字典）。
    """
    try:
        # 使用 pandas 读取 .xlsx 文件
        data = pd.read_excel(file_path)

        # 将 DataFrame 转换为字典
        data_dict = data.to_dict(orient='index')

        # 返回包含信息的字典
        return data_dict
    except Exception as e:
        print(f"读取 .xlsx 文件时发生错误：{e}")
        return None
    
def Read_BCF_SSL(data_1_file):
    train_num = 0
    train_SSL_data = []
    for sess in os.listdir(data_1_file):
        file_dir = os.path.join(data_1_file, sess)
        wavname = file_dir.split("/")[-1][:-4]
        #print(wavname)
        audio_input, sample_rate = process_wav_file(file_dir,3)
        #audio_input, sample_rate = sf.read(filename)
        input_values = processor(audio_input, sampling_rate=sample_rate,
                                    return_tensors="pt").input_values
        # training set
        one_mel_data = {}
        one_mel_data['id'] = wavname
        one_mel_data['wav_encodings'] = input_values
        train_SSL_data.append(one_mel_data)
        train_num = train_num + 1
    return train_SSL_data

def combine(data,label):
    num = 0
    for i in range(len(data)):
        for j in range(len(label)):
            if(data[i]['id'] == label[j]['Utterance']):
                label[j]['wav_encodings'] = [data[i]['wav_encodings']]
                num = num + 1
    print(num)
    return label

def Get_data(Input_data):
    name_list = ['1_Bright','2_Dark','3_High','4_Low','5_Strong','6_Weak','7_Calm','8_Unstable','9_Well-modulated','10_Monotonous','11_Heavy','12_Clear','13_Noisy','14_Quiet','15_Sharp','16_Fast','17_Slow']
    for i in range(len(Input_data)):
        data_1 = []
        for key in Input_data[i]:
            if(key in name_list):
                data_1.append(Input_data[i][key])
        Input_data[i]['adjective'] = np.array(data_1).reshape(-1, 1)
    Input_data = STD(Input_data,'adjective')
    for i in range(len(Input_data)):
        Input_data[i]['1_Bright'] = Input_data[i]['adjective'][0][0]
        Input_data[i]['2_Dark'] = Input_data[i]['adjective'][1][0]
        Input_data[i]['3_High'] = Input_data[i]['adjective'][2][0]
        Input_data[i]['4_Low'] = Input_data[i]['adjective'][3][0]
        Input_data[i]['5_Strong'] = Input_data[i]['adjective'][4][0]
        Input_data[i]['6_Weak'] = Input_data[i]['adjective'][5][0]
        Input_data[i]['7_Calm'] = Input_data[i]['adjective'][6][0]
        Input_data[i]['8_Unstable'] = Input_data[i]['adjective'][7][0]
        Input_data[i]['9_Well-modulated'] = Input_data[i]['adjective'][8][0]
        Input_data[i]['10_Monotonous'] = Input_data[i]['adjective'][9][0]
        Input_data[i]['11_Heavy'] = Input_data[i]['adjective'][10][0]
        Input_data[i]['12_Clear'] = Input_data[i]['adjective'][11][0]
        Input_data[i]['13_Noisy'] = Input_data[i]['adjective'][12][0]
        Input_data[i]['14_Quiet'] = Input_data[i]['adjective'][13][0]
        Input_data[i]['15_Sharp'] = Input_data[i]['adjective'][14][0]
        Input_data[i]['16_Fast'] = Input_data[i]['adjective'][15][0]
        Input_data[i]['17_Slow'] = Input_data[i]['adjective'][16][0]
    return Input_data

def Get_Train_data(data):
    print(data[0])
    empty_lists = [[] for _ in range(10)]
    for i in range(len(data)):
        empty_lists[data[i]['speaker_idx']-1].append(data[i])
    return empty_lists





if __name__=="__main__":
    #downsample_folder(data_file,data_1_file,16000)
    SSL_Data = Read_BCF_SSL(data_1_file)
    SSL_Label = read_excel_file(label_file)
    Com_train_data = combine(SSL_Data,SSL_Label)
    Final_Train_data = Get_data(Com_train_data)
    Train_data = Get_Train_data(Final_Train_data)



    file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Deep3Layer/10-Flod/Deep3Layer_Hubert/Speech_data.pickle', 'wb')
    pickle.dump(Train_data, file)
    file.close()