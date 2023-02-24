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


from statsmodels.tsa.api import Holt

y_hat_avg = test.copy()
fit = Holt(np.asarray(train['Count'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
y_hat_avg['Holt_linear'] = fit.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['Holt_linear']))
print(rms)

