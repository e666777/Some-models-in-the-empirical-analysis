import pandas as pd #pands别名pd
df = pd.read_csv('D//Project/python/##Example##/时间序列分析/train.csv') #读取csv文件
print(df.head(5)) #读取前5行数据

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv',nrows=11856)
train=df[0:10392]#训练集
test=df[10392:]#测试集
df['Timestamp']=pd.to_datetime(df['Datetime'],format="%d-%m-%Y %H:%M")
df.index=df['Timestamp']  #在train文档中新增加一列Timestamp
df=df.resample('D').mean() #按天采样，计算均值

train['Timestamp']=pd.to_datetime(train['Datetime'],format="%d-%m-%Y %H:%M")
train.index=train['Timestamp']
train=train.resample('D').mean()

test['Timestamp']=pd.to_datetime(test['Datetime'],format="%d-%m-%Y %H:%M")
test.index=test['Timestamp']
test=test.resample('D').mean()

#数据可视化
train.Count.plot(figsize=(15,7),title='Daily xxx',fontsize=14)
test.Count.plot(figsize=(15,7),title='Daily xxx',fontsize=14)
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
#
dd=np.asarray(train['Count']) #将train中的Count列转化为array(数组)形式
y_hat=test.copy()
y_hat['naive']=dd[len(dd)-1] #预测数据均为test的最后一个数值
plt.figure(figsize=(12,8))
plt.plot(train.index,train['Count'],label='Train')
plt.plot(test.index,test['Count'],label='Test')
plt.plot(y_hat.index,y_hat['naive'],label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()
#均方根误差计算
rms = sqrt(mean_squared_error(test['Count'], y_hat['naive']))
print(rms)
