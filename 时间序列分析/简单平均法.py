import pandas as pd #pands别名pd
df = pd.read_csv('train.csv') #读取csv文件
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


y_hat_avg=test.copy()
y_hat_avg['avg_forecast']=train['Count'].mean() #取train中Count列的平均值
plt.figure(figsize=(12,8))
plt.plot(train['Count'],label='Train')
plt.plot(test['Count'],label='Test')
plt.plot(y_hat_avg['avg_forecast'],label='Average Forecast')
plt.legend(loc='best') #对性能图做出一些补充说明的问题
plt.show()

rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['avg_forecast']))
print(rms)
