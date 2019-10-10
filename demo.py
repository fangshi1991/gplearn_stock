# encoding:utf-8
from load_data.loader import load_day_data,load_minute_data
import numpy as np
import matplotlib.pyplot as plt
from gplearn.utils import check_random_state
from gplearn.genetic import SymbolicRegressor
import time 
import numba 
import copy 
import h5py
import gcdata_chuli
from data_processing.DataProcessing import data_chuli


# 定义一个函数，用于进行遗传进化    
if __name__ == '__main__':
    '''
    由于数据可能需要多次处理，这里现将处理好的数据进行保存
    '''
    
    '''
    stocks = list(pd.read_csv('C:\\Users\\94006\\Desktop\\HS300.csv')['HS300'])
    start_time = '2010-06-30'
    end_time= '2018-12-30'
    stock_list = [str(i).zfill(6) for i in stocks]
    day_or_minute = 'day'
    metric = 'stock_dedicated'
    
    train_ratio = 0.7
    adjust=True
    
    data_all = data_chuli(start_time = start_time,
                            end_time = end_time,
                            stock_list = stock_list,
                            day_or_minute = day_or_minute,
                            train_ratio = train_ratio,
                            adjust = adjust)
    (x_train,y_train,x_test,y_test) = data_all.time_adjust()#获取训练和测试用的数据
    x_train_HS300 = 'D:\\HS300_x_train_datas150.npy'
    y_train_HS300 = 'D:\\HS300_y_train_datas150.npy'
    x_test_HS300 = 'D:\\HS300_x_test_datas150.npy'
    y_test_HS300 = 'D:\\HS300_y_test_datas150.npy'
    np.save(x_train_HS300,np.array(x_train))
    np.save(y_train_HS300,np.array(y_train))
    np.save(x_test_HS300,np.array(x_test))
    np.save(y_test_HS300,np.array(y_test))
    
    '''
   
    x_train = np.array(np.load('D:\\HS300_x_train_datas150.npy')) # 不同用户数据保存的位置可能不一样，请根据自身的地址进行调整
    y_train = np.array(np.load('D:\\HS300_y_train_datas150.npy'))
    x_test = np.array(np.load('D:\\HS300_x_test_datas150.npy'))
    y_test =np.array( np.load('D:\\HS300_y_test_datas150.npy'))
    (a,b,c,d) = jinghua_data(x_train,y_train,x_test,y_test)
 
    print (u'数据准备完成，进入进化')
    est_gp = SymbolicRegressor(population_size=5000,
                           generations=8, stopping_criteria=10000,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0,
                           metric= 'stock_dedicated',             # 本参数为 = ‘stock_dedicated’则为股票的
                           n_jobs=2)# 构建一个遗传进化的类 
    print (u'类构件完成')
    input()                        
    x_trains = a
    y_trains = b
    est_gp.fit(x_trains, y_trains)   
