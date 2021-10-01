import numpy as np
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.svm import LinearSVC
from ensemble.ensembles import BinaryRelevance, ClassifierChains, CalibratedLabelRanking, RandomKLabelsets, MLKNN
#from multi_code.tree.ml_dt import MLDecisionTree
# from mlclas.neural import BPMLL
# from multi_code.svm.ranking_svm import RankingSVM
# from mlclas.stats import UniversalMetrics

# files = ['datasets/scene_train', 'datasets/scene_test']
#
# # load files
# data = datasets.load_svmlight_files(files, multilabel=True)
# train_data = data[0]
# train_target = np.array(MultiLabelBinarizer().fit_transform(data[1]))
# test_data = data[2]
# test_target = data[3]
#
# # feature extraction using PCA
# feature_size = train_data.shape[1]
# pca = PCA(n_components=(feature_size * 10) // 100)
# train_data_trans = csr_matrix(pca.fit_transform(train_data.todense()))
# test_data_trans = csr_matrix(pca.transform(test_data.todense()))
#
# """
#     train and predict using any of following scripts:
#
#     1.  result = BinaryRelevance(LinearSVC()).fit(train_data, train_target).predict(test_data)
#
#     2.  result = ClassifierChains(LinearSVC()).fit(train_data, train_target).predict(test_data)
#
#     3.  result = CalibratedLabelRanking(LinearSVC()).fit(train_data, train_target).predict(test_data)
#
#     4.  result = RandomKLabelsets(LinearSVC()).fit(train_data, train_target).predict(test_data)
#
#     5.  result = MLKNN(any integer, for example 6).fit(train_data, train_target).predict(test_data)
#
#     6.  result = MLDecisionTree(min_num=10).fit(train_data_trans, train_target).predict(test_data_trans)
#
#     7.  result = BPMLL(print_procedure=True, neural=0.4, regularization=0, epoch=40, normalize='max').fit(train_data_trans, train_target)
#                 .predict(test_data_trans)
#
#     8.  result = RankingSVM(normalize='l2', print_procedure=True).fit(train_data_trans, train_target, 8).predict(test_data_trans)
#
# """
# # result = ClassifierChains(LinearSVC()).fit(train_data, train_target).predict(test_data)
# #
# # # metrics
# # m = UniversalMetrics(test_target, result)
# # print('precision: ' + str(m.precision()))
# # print('accuracy: ' + str(m.accuracy()))
# # result1 = MLDecisionTree(min_num=10).fit(train_data_trans, train_target).predict(test_data_trans)
# # result2 = RankingSVM(normalize='l2', print_procedure=True).fit(train_data_trans, train_target, 8).predict(test_data_trans)
# # print(result1)
# # print(result2)
# def eliminate_data(data_x, data_y):
#     data_num = data_y.shape[0]
#     label_num = data_y.shape[1]
#     full_true = np.ones(label_num)
#     full_false = np.zeros(label_num)
#
#     i = 0
#     while(i < len(data_y)):
#         if (data_y[i] == full_true).all() or (data_y[i] == full_false).all():
#             data_y = np.delete(data_y, i, axis=0)
#             data_x = np.delete(data_x, i, axis=0)
#         else:
#             i = i + 1
#
#     return data_x, data_y
# def load_data(dataset_name):
#     x_train = np.load(dataset_name + '\\x_train.npy')
#     y_train = np.load(dataset_name + '\\y_train.npy')
#     x_test = np.load(dataset_name + '\\x_test.npy')
#     y_test = np.load(dataset_name + '\\y_test.npy')
#     x_train, y_train = eliminate_data(x_train, y_train)
#     x_test, y_test = eliminate_data(x_test, y_test)
#     return x_train, y_train, x_test, y_test
#
# x_train, y_train, x_test, y_test = load_data('C:\\Users\\hexiong\\PycharmProjects\\multi1\\multi_code\\datasets\\yeast')
# print(x_train.shape)
# print(y_train.shape)
# # print(type(x_train))
# # print(x_train[0])
# x1 = []
# x2 = []
# for line in x_train:
#     x1.append(list(line))
# for item in x_test:
#     x2.append(list(item))
# x_train = csr_matrix(x1)
# x_test = csr_matrix(x2)
# #result = MLKNN(6).fit(train_data, train_target).predict(test_data)
# result,result1,result2 = MLKNN(6).fit(x_train, y_train).predict(x_test)
# c = 0
# y_test2 = []
# print(result)
# for i in range(len(y_test)):
#     y_test2.append(np.where(y_test[i]==1)[0])
#     s = set(np.where(y_test[i]==1)[0])
#     if result2[i] in s:
#         c+=1
# print(c)
# print(len(y_test))
#
# from multi_code.pingjia import pingjia
# r1 = pingjia(y_test2,result1,result2,result,len(y_test[0]))
# print("hamming_loss:",r1.hamming_loss())
# print("one_error:",r1.one_error())
# print("coverage:",r1.coverage())
# print("ranking_loss:",r1.ranking_loss())
# print("average_precision:",r1.average_precision())

import scipy.io as scio

def load_data():
    data = scio.loadmat('C:\\Users\\hexiong\\PycharmProjects\\multi1\\multi_code\\datasets\\data2.mat')
    #data = scio.loadmat('data2.mat')
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    x_lable = data['x_lable']
    y_lable = data['zhize']
    test_key = data['test_key']
    return x_train,x_test,y_train,y_test,test_key,y_lable,x_lable
def data_hand(x_train,x_test):#对输入的训练数据和测试数据进行降维
    pca = PCA(200)
    new_xtrain = pca.fit_transform(x_train)
    new_xtest = pca.transform(x_test)
    return new_xtrain,new_xtest
x_train,x_test,y_train,y_test,test_key,y_lable,x_lable = load_data()
x_train,x_test = data_hand(x_train,x_test)
x_train = csr_matrix(x_train)
x_test = csr_matrix(x_test)
result,result1,result2 = MLKNN(6).fit(x_train, y_train).predict(x_test)
scio.savemat("mlknn_result.mat",{'result':result})
from pingjia import pingjia
y_test2 = []
for i in range(len(y_test)):
    y_test2.append(np.where(y_test[i]==1)[0])

r1 = pingjia(y_test2,result1,result2,result,len(y_test[0]))
print("hamming_loss:",r1.hamming_loss())
print("one_error:",r1.one_error())
print("coverage:",r1.coverage())
print("ranking_loss:",r1.ranking_loss())
print("average_precision:",r1.average_precision())