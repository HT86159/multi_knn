import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import copy
def load_data():
    data = scio.loadmat('data2.mat')
    # data = scio.loadmat('data2.mat')
    x_test = data['x_test']
    y_test = data['y_test']
    ourmodel_data  = scio.loadmat("ourmodel.mat")['result']
    mlknn_data = scio.loadmat("mlknn_result.mat")['result']
    mldt_data = scio.loadmat("mldt_result.mat")['result']
    adaboost_mh_data = scio.loadmat("adaboost_mh_result.mat")['result']
    bpml_data = scio.loadmat("bpml_result.mat")['result']
    return y_test,ourmodel_data,mlknn_data,mldt_data,adaboost_mh_data,bpml_data
def compute_accuracy(y_pre,v_ys,num):
    y_pre2 = []
    for i in range(len(v_ys)):
        for j in range(num):
            s = np.argmax(y_pre[i])
            y_pre2[i][s] = 1.0
            y_pre[i][s] = 0.0
    print("ypre2 shape", y_pre.shape)
    s1 = y_pre + v_ys
    p1 = 0  # accuracy
    p2 = 0  # recall
    n11 = 0
    n22 = 0
    n33 = 0
    data_list = []
    for i in range(len(s1)):
        n1 = sum(s1[i] == 2)
        n2 = sum(v_ys[i] == 1)
        n3 = sum(y_pre[i] == 1)

        p1 = p1 + n1 / n2
        p2 = p2 + n1 / (n3+0.001)
        n11 += n1
        n22 += n2
        n33 += n3

    p1 = p1 / len(y_pre)
    p2 = p2 / len(y_pre)
    print("recall:", p1)
    print("precision:", p2)
    print("fugai:", n11)
    print("labels number", n22)
    print("predict number ", n33)
    #f1.close()
    return p1,p2
y_test,ourmodel_data,mlknn_data,mldt_data,adaboost_mh_data,bpml_data = load_data()
y11 = []
y12 = []
y21 = []
y22 = []
y31 = []
y32 = []
y41 = []
y42 = []
y51 = []
y52 = []
x = []
for i in range(1,13):
    y_pre_s1 = copy.deepcopy(ourmodel_data)
    y_pre_s2 = copy.deepcopy(mlknn_data)
    y_pre_s3 = copy.deepcopy(mldt_data)
    y_pre_s4 = copy.deepcopy(adaboost_mh_data)
    y_pre_s5 = copy.deepcopy(bpml_data)

    p1,p2 = compute_accuracy(y_pre_s1,y_test,i)
    y11.append(p1)
    y12.append(p2)
    p1, p2 = compute_accuracy(y_pre_s2, y_test, i)
    y21.append(p1)
    y22.append(p2)
    p1, p2 = compute_accuracy(y_pre_s3, y_test, i)
    y31.append(p1)
    y32.append(p2)
    p1, p2 = compute_accuracy(y_pre_s4, y_test, i)
    y41.append(p1)
    y42.append(p2)
    p1, p2 = compute_accuracy(y_pre_s5, y_test, i)
    y51.append(p1)
    y52.append(p2)
plt.title('recall')
plt.plot(x,y11,'r-',label = 'ourmodel')
plt.plot(x,y21,'b-',label = 'mlknn')
plt.plot(x,y31,'y-',label = 'mldt')
plt.plot(x,y41,'g-',label = 'adaboost.mh')
plt.plot(x,y51,'p-',label = 'bpml')
plt.xticks(x)
plt.legend()
plt.ylabel('value')
plt.xlabel('top k label')
plt.savefig('com1.jpg')
plt.show()
plt.title('precision')
plt.plot(x,y12,'r-',label = 'ourmodel')
plt.plot(x,y22,'b-',label = 'mlknn')
plt.plot(x,y32,'y-',label = 'mldt')
plt.plot(x,y42,'g-',label = 'adaboost.mh')
plt.plot(x,y52,'p-',label = 'bpml')
plt.xticks(x)
plt.legend()
plt.ylabel('value')
plt.xlabel('top k label')
plt.savefig('com2.jpg')
plt.show()