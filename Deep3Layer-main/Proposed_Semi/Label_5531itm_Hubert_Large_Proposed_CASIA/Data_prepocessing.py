#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Jan  9 20:32:28 2018

@author: shixiaohan
"""
import re
import wave
import numpy as np
import python_speech_features as ps
import soundfile as sf
import os
import glob
import pickle
import torch
import torchaudio

rootdir = '/mnt/data1/liyongwei/Database/CASIA/相同文本300/'

from transformers import AutoProcessor, HubertModel

processor = AutoProcessor.from_pretrained("/mnt/data1/liyongwei/SSL_Models/facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("/mnt/data1/liyongwei/SSL_Models/facebook/hubert-large-ls960-ft")   # 用于提取通用特征，768维

label_list = [0, 1, 2, 3, 4, 5]


def emo_change(x):
    if x == 'angry':
        x = 0
    if x == 'fear':
        x = 1
    if x == 'happy':
        x = 2
    if x == 'neutral':
        x = 3
    if x == 'sad':
        x = 4
    if x == 'surprise':
        x = 5
    return x

def load_data(dir):
    with open(dir, 'rb') as file:
        data = pickle.load(file)
    return data

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

def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    channel = file.getnchannels()
    sampwidth = file.getsampwidth()
    framerate = file.getframerate()
    frames = file.getnframes()
    duration = frames/framerate
    wav_length = 3 * framerate
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    file.close()
    return wavedata, time, framerate

def Read_CASIA_Spec():
    filter_num = 40
    train_num = 0
    train_mel_data = []
    for speaker in os.listdir(rootdir):
        sub_dir = os.path.join(rootdir, speaker)
        for emo in os.listdir(sub_dir):
            sub_dir_1 = os.path.join(sub_dir, emo, '*.wav')
            files = glob.glob(sub_dir_1)
            for filename in files:
                wavname = filename.split("/")[-1][:-4]
                data, time, rate = read_file(filename)
                mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
                # training set
                mel_data = []
                one_mel_data = {}
                part = mel_spec
                delta1 = ps.delta(mel_spec, 2)
                delta2 = ps.delta(delta1, 2)
                input_data_1 = np.concatenate((part, delta1), axis=1)
                input_data = np.concatenate((input_data_1, delta2), axis=1)
                mel_data.append(input_data)
                one_mel_data['id'] = speaker + '_' + emo + '_' + wavname
                mel_data = np.array(mel_data)
                one_mel_data['spec_data'] = mel_data
                one_mel_data['label'] = emo
                one_mel_data['speaker'] = speaker
                train_mel_data.append(one_mel_data)
                train_num = train_num + 1
    #print(train_num)
    return train_mel_data

def Read_CASIA_Trad():
    filter_num = 40
    train_num = 0
    train_mel_data = []
    for speaker in os.listdir(rootdir):
        sub_dir = os.path.join(rootdir, speaker)
        for emo in os.listdir(sub_dir):
            sub_dir_1 = os.path.join(sub_dir, emo, '*.wav')
            files = glob.glob(sub_dir_1)
            for filename in files:
                wavname = filename.split("/")[-1][:-4]
                audio_input, sample_rate = process_wav_file(filename,3)
                #audio_input, sample_rate = sf.read(filename)
                input_values = processor(audio_input, sampling_rate=sample_rate,
                                            return_tensors="pt").input_values
                one_mel_data = {}
                one_mel_data['id'] = speaker + '_' + emo + '_' + wavname
                one_mel_data['wav_encodings'] = input_values
                train_mel_data.append(one_mel_data)
                train_num = train_num + 1
    #print(train_num)
    return train_mel_data

def Read_IEMOCAP_Three_Layer():
    Fine_dir = '/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Semi/'
    Each_name = ['SSL_Data2vec_CASIA']
    name_list = ['1_Bright','2_Dark','3_High','4_Low','5_Strong','6_Weak', \
                 '7_Calm','8_Unstable','9_Well-modulated','10_Monotonous',\
                 '11_Heavy','12_Clear','13_Noisy','14_Quiet','15_Sharp',\
                    '16_Fast','17_Slow']
    All_group = []
    for itm in Each_name:
        Itm_dir = Fine_dir + itm 
        data_11 = []
        for iem_1 in name_list:
            One_gro = {}
            fin_dir = Itm_dir + '/CASIA_result_' + iem_1 + '.pickle'
            data_iem = load_data(fin_dir)
            One_gro['id'] = iem_1
            One_gro['data'] = data_iem          
            data_11.append(One_gro)
        one_fin_data = []
        for i in range(len(data_11[0]['data'])):
            one_fin = {}
            one_fin['id'] = data_11[0]['data'][i]['id']
            one_fin['1_Bright'] = data_11[0]['data'][i]['Predict_label'][0]
            one_fin['2_Dark'] = data_11[1]['data'][i]['Predict_label'][0]
            one_fin['3_High'] = data_11[2]['data'][i]['Predict_label'][0]
            one_fin['4_Low'] = data_11[3]['data'][i]['Predict_label'][0]
            one_fin['5_Strong'] = data_11[4]['data'][i]['Predict_label'][0]
            one_fin['6_Weak'] = data_11[5]['data'][i]['Predict_label'][0]
            one_fin['7_Calm'] = data_11[6]['data'][i]['Predict_label'][0]
            one_fin['8_Unstable'] = data_11[7]['data'][i]['Predict_label'][0]
            one_fin['9_Well-modulated'] = data_11[8]['data'][i]['Predict_label'][0]
            one_fin['10_Monotonous'] = data_11[9]['data'][i]['Predict_label'][0]
            one_fin['11_Heavy'] = data_11[10]['data'][i]['Predict_label'][0]
            one_fin['12_Clear'] = data_11[11]['data'][i]['Predict_label'][0]
            one_fin['13_Noisy'] = data_11[12]['data'][i]['Predict_label'][0]
            one_fin['14_Quiet'] = data_11[13]['data'][i]['Predict_label'][0]
            one_fin['15_Sharp'] = data_11[14]['data'][i]['Predict_label'][0]
            one_fin['16_Fast'] = data_11[15]['data'][i]['Predict_label'][0]
            one_fin['17_Slow'] = data_11[16]['data'][i]['Predict_label'][0]
            exclude_key = 'id'  # 要排除的键
            values_list = [value for key, value in one_fin.items() if key != exclude_key]
            one_fin['ALL_Vec'] = values_list
            one_fin_data.append(one_fin)
        All_group.append(one_fin_data)
    return All_group

def Seg_IEMOCAP(train_data_spec_1,train_data_spec,train_data_trad):
    for i in range(len(train_data_spec)):
        for j in range(len(train_data_trad)):
            if (train_data_spec[i]['id'] == train_data_trad[j]['id']):
                train_data_spec[i]['wav_encodings'] = train_data_trad[j]['wav_encodings']

    for i in range(len(train_data_spec)):
        for j in range(len(train_data_spec_1)):
            if (train_data_spec[i]['id'] == train_data_spec_1[j]['id']):
                train_data_spec[i]['ALL_Vec'] = train_data_spec_1[j]['ALL_Vec']

    num = 0
    train_data_map = []
    for i in range(len(train_data_spec)):
        if (len(train_data_spec[i]) == 6):
            train_data_map.append(train_data_spec[i])
            num = num + 1
    print(num)
    return train_data_map

def Train_data(train_map_pre):

    train_map = []
    num = 0
    for i in range(len(train_map_pre)):
        data = {}
        data['label'] = emo_change(train_map_pre[i]['label'])
        data['wav_encodings'] = train_map_pre[i]['wav_encodings']
        data['spec_data'] = np.array(train_map_pre[i]['ALL_Vec']).reshape(-1, 1)
        data['id'] = train_map_pre[i]['id']
        data['speaker'] = train_map_pre[i]['speaker']
        if(data['label'] in label_list):
            data['label'] = data['label']
            train_map.append(data)
            num = num + 1

    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []

    for i in range(len(train_map)):
        if (train_map[i]['speaker']== 'liuchanhg'):
            data_1.append(train_map[i])
        if (train_map[i]['speaker'] == 'wangzhe'):
            data_2.append(train_map[i])
        if (train_map[i]['speaker']== 'zhaoquanyin'):
            data_3.append(train_map[i])
        if (train_map[i]['speaker'] == 'ZhaoZuoxiang'):
            data_4.append(train_map[i])

    print(len(data_1))
    print(len(data_2))
    print(len(data_3))
    print(len(data_4))
    data = []
    data.append(data_1)
    data.append(data_2)
    data.append(data_3)
    data.append(data_4)
    return data
if __name__ == '__main__':

    train_data_thre = Read_IEMOCAP_Three_Layer()  
    train_data_spec = Read_CASIA_Spec()
    train_data_trad = Read_CASIA_Trad()
    train_data_map = Seg_IEMOCAP(train_data_thre[0],train_data_spec,train_data_trad)
    print(len(train_data_map))
    #print(train_data_map[0][0])
    Train_data = Train_data(train_data_map)
    file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Semi/Hubert_Large_Proposed_CASIA/Train_data.pickle', 'wb')
    pickle.dump(Train_data, file)
    file.close()