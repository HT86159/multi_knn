import pandas as pd
import numpy as np
oridata=pd.read_excel (r'C:\Users\86159\Desktop\我的\科研与竞赛\大创\2021-2022\九月\20210831_医案数据.xlsx')

oridata=oridata .iloc[:60,:]
oridata.to_excel('multi_code\datasets\data_all.xlsx')
'''
data=pd.read_excel ('multi_code\datasets\data_all.xlsx')
data=data.iloc[:100,:]
sym=data.iloc[:,2]
med=data.iloc[:,3]
str=sym.iloc[-1]
str1=med.iloc[-1]
for i in range(len(sym)-1):
    str1=str1+';'+med.iloc[i]
    str=str+';'+sym.iloc[i]
symlis=str.split(';')
medlis=str1.split(';')
print('------symlis------')
#print(symlis)
print('------medlis------')
#print(medlis)

symlis=sorted(set(symlis))
medlis=sorted(set(medlis))
symdic=dict(zip(symlis,range(len(symlis))))
print('------symdic------')
print(symdic)
meddic=dict(zip(medlis,range(len(medlis))))
print('------meddic-----')
print(meddic)


print(len(data),len(symdic ))
symmat=np.zeros((len(data),len(symdic )))
for i in range(len(sym)):
    symf1=sym[i].split(';')
    for j in symf1:
        symmat[i][symdic[j]]=1
np.savetxt('multi_code\datasets\symmat.txt',symmat)

medmat=np.zeros((len(data),len(meddic )))
for i in range(len(med)):
    symf1=med[i].split(';')
    for j in symf1:
        medmat[i][meddic[j]]=1
np.savetxt('multi_code\datasets\medmat.txt',medmat)

print('----------done!!!!!-----------')'''
