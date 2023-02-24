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


warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

import statsmodels.api as sm

y_hat_avg=test.copy()
fit=sm.tsa.statespace.SARIMAX(train.Count,order=(2,1,4),seasonal_order=(0,1,1,7)).fit()
y_hat_avg['SARIMA']=fit.predict(start="2013-11-1",end="2013-12-31",dynamic=True)
plt.figure(figsize=(16,8))
plt.plot(train['Count'],label='Train')
plt.plot(test['Count'],label='Test')
plt.plot(y_hat_avg['SARIMA'],label='SARIMA')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['SARIMA']))
print(rms)
