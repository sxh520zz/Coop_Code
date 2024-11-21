#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:10:17 2019

@author: jdang03
"""

import pickle
import numpy as np 
from sklearn import metrics
from sklearn.metrics import mean_squared_error





def concordance_correlation_coefficient(y_true,Y_pred,
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


def main(item):
    dir = '/mnt/data1/liyongwei/Project/Xiaohan_code/Deep3Layer/Proposed_Semi/SSL_Hubert/Final_result_' + item + '.pickle'
    with open(dir, 'rb') as file:
        final_result =pickle.load(file)            
    true_label = []    
    predict_label = []   
    for i in range(len(final_result)):
        for j in range(len(final_result[i])):
            predict_label.append(final_result[i][j]['Predict_label'])
            true_label.append(final_result[i][j]['True_label'])

    accuracy_recall = concordance_correlation_coefficient(true_label, predict_label)
    accuracy_f1 = mean_squared_error(true_label, predict_label)
    print(f"The result for {item} is: {accuracy_recall:.4f}/{accuracy_f1:.4f}")


if __name__ == "__main__":

    name_list = ['1_Bright','2_Dark','3_High','4_Low','5_Strong','6_Weak','7_Calm','8_Unstable','9_Well-modulated','10_Monotonous','11_Heavy','12_Clear','13_Noisy','14_Quiet','15_Sharp','16_Fast','17_Slow']
    
    for item in name_list:
        main(item)
