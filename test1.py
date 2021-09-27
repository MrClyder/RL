import pandas as pd
import numpy as np
# from data_clean import *
#
# source_path = '/home/dl/Desktop/data/ETH_swap_data/data'
# time_interval = 1000 #10s.  #60s = 60000
# save_path = '/home/dl/ren/transformer-prediction/data/'  ##保存的数据路径，自定义
# df = factor_data(source_path,time_interval,save_path)

# train_data = pd.read_csv('/home/dl/ren/transformer-prediction/7.24_7.27_eth_swap_150s_data.csv', index_col=[0])
# print(train_data)
# train_data = train_data.drop(columns=('updatetime','day'))
# print(train_data)
# sample = 2000
# # test_data = np.array(train_data)
# test_data = np.array(train_data)
# val_data = np.array(train_data)
# # print(train_data(:sample))
# test_data = test_data(:sample,0)
# val_data = val_data(sample:)
# print(test_data)
# print(len(val_data))
#


from scipy.optimize import minimize
import numpy as np
e = 1e-10

number = []
for i in range(2):
    number.append(i)

def cons1(a):
    return {'type': 'ineq', 'fun': lambda x:x[a]-e}
def cons2(a):
    return {'type': 'ineq', 'fun': lambda x:1-x[a]}



def zeros(n):  #给定变量初值
    x0=[]
    for i in range(n):
        x0.append(0)  #这里假设给定各变量初值为零
    return x0


# fun = lambda x: np.sqrt(((-2769*x[3]+10840)/65000)**2+((5161*x[3]-6888)/130000)**2+((91*x[3]-2448)/32500)**2+ \
#                         ((-327*x[2]+9516)/10000)**2+((313*x[2]-2024)/10000)**2+((15*x[2]+8)/10000)**2+ \
#                         ((-223*x[1]+6675)/10000)**2+((217*x[1]+323)/10000)**2+((5*x[1]+3)/10000)**2+ \
#                         ((-57*x[0]+4918)/5000)**2+((57*x[0]+82)/5000)**2)
fun = lambda x: np.sqrt(((93*x[1]-104)/400)**2+((-33*x[1]+34)/200)**2+((-27*x[1]+36)/400)**2+ \
                        ((3*x[0]-1)/20)**2+((-3*x[0]+1)/20)**2)
n=2
x0 = zeros(n)
print(x0)
a = list(map(cons1,number))
b = list(map(cons2,number))
c = a+b
res = minimize(fun,x0,method='SLSQP',constraints=c)
print(res)
print('best', res.x)
