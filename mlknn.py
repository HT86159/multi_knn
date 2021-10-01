# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 11:35
# @Author  : dx
# @FileName: mlknn.py
# @Function:
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from multi_code.ensemble.ensembles import MLKNN
from sklearn.decomposition import PCA
from multi_code.pingjia import pingjia

import scipy.io as scio

def load_data():
    sym_data = np.loadtxt('multi_code\datasets\symmat.txt')
    x_train = sym_data[:80]
    x_test = sym_data[80:]
    med_data=np.loadtxt ('multi_code\datasets\medmat.txt')
    y_train = med_data[:80]
    y_test = med_data[80:]
    #x_label = data['x_lable']
    #y_label = data['zhize']
    #test_key = data['test_key']
    return x_train, x_test, y_train, y_test#, test_key, y_label, x_label

def data_hand(x_train, x_test):#对输入的训练数据和测试数据进行降维
    pca = PCA(80)
    new_xtrain = pca.fit_transform(x_train)#训练模型，也就是模型已经确定，同时返回降维的值
    new_xtest = pca.transform(x_test)##用已经训练好的数据进行降维
    return new_xtrain, new_xtest

def process():
    x_train,x_test,y_train,y_test= load_data()#load the data,,,,test_key,y_lable,x_lable
    x_train,x_test = data_hand(x_train,x_test)#用PCA进行降维
    x_train = csr_matrix(x_train)#压缩成csr格式，方便计算
    x_test = csr_matrix(x_test)
    result,result1,result2 = MLKNN(10).fit(x_train, y_train).predict(x_test)
    scio.savemat("mlknn_result.mat",{'result':result})
    result1=pd.DataFrame (result1)
    result1 =result1 .values
    np.savetxt('multi_code\datasets\predict.txt',result1 )

    y_test2 = []
    for i in range(len(y_test)):
        y_test2.append(np.where(y_test[i] == 1)[0])
    r1 = pingjia(y_test2, result1, result2, result, len(y_test[0]))
    print("hamming_loss:", r1.hamming_loss())
    print("one_error:", r1.one_error())
    print("coverage:", r1.coverage())
    print("ranking_loss:", r1.ranking_loss())
    print("average_precision:", r1.average_precision())

if __name__ == '__main__':
    process()