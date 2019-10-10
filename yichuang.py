# encoding:utf-8
from loader import load_day_data,load_minute_data
import numpy as np
import matplotlib.pyplot as plt
from gplearn.utils import check_random_state
from gplearn.genetic import SymbolicRegressor
import time 
import numba 
import copy 
import h5py
import gc

# ['open', 'high', 'low', 'close', 'volume', 'price', 'change', 'mean_five', 'highest_five', 'lowest_five',
# 'mean_20', 'highest_20', 'lowest_20', '20_profit', '20_mean', '5_mean', '20_mean_vol', '5_mean_vol']
from collections import OrderedDict
time_now = time.time()
import pandas as pd
# 定义一个函数，用于计算最大回撤 input()
# 最大回撤（输入为list或array）
@numba.jit
def MaxDrawdown(strategy): 
    #print (strategy)
    length = (strategy.shape)[0]
    drawdown = []
    py =strategy[0]
    for i in range(1,length):
        px = strategy[i]
        py = max(strategy[:i])
        drawdown.append(1-(px/py))
    if max(drawdown)>0:
        return max(drawdown)
    else:
        return 0
    
#定义一个函数，判断是否存在np.nan
@numba.jit
def kong_pan(xulie):
    for i in xulie:
        if abs(i)>=0:
            pass
        else:
            return True      
    return False

# 定义一个类，用于进行数据方面的处理
class data_chuli(object):
    def __init__(self, start_time,end_time, stock_list,day_or_minute = 'day',train_ratio = 0.3,adjust=True):
        self.start_time = start_time
        self.end_time = end_time 
        self.stock_list = stock_list        # 股票列表
        self.day_or_minute = day_or_minute  # 交易日历
        self.train_ratio = train_ratio      # 测试数据占总体数据的比例
        self.adjust = adjust                # 是否复权
     
    
    def data_get(self):
        '''
        本程序主要用于获取股票数据
        '''
        print (self.day_or_minute)
        if self.day_or_minute =='day':
            print (u'获取日线数据')
            [data,startdate,enddate] = load_day_data(stockList = self.stock_list,
                                                    start = self.start_time,
                                                    end = self.end_time)
        else:
            print (u'获取分钟数据')
            [data,startdate,enddate] = load_minute_data(stockList = self.stock_list,
                                                    start = self.start_time,
                                                    end = self.end_time)
        return data 
        
    
    def index_get(self,indexes = '000001'):
        data = load_day_data(indexes=indexes,start = self.start_time,end = self.end_time)
        
        #print (data[indexes].head())
        return data[indexes]
        
    def adaptability_compute(self,cycle = 5):
        
        '''
        本程序主要是进行适应度计算
        本处主要计算收益与最大回撤的比值
        '''
        data  =self.data_get()
        new_data = OrderedDict()
        for stocks in self.stock_list:
            stock = data[stocks]
            #print (stock.head(10))
            closes = list(stock['close']) 
            close_list = []
            try:
                for i in range(cycle):
                    close_list.append(closes[i+1:]+[closes[-1] for t in range(i+1)])
                close_list = np.array(close_list).T    
                t = stock.shift(-1*cycle)['close']/stock['close']
                maxdowntown = [MaxDrawdown(close_list[i,:]) for i in range(len(close_list[:,0]))]
                stock['after_5_profit'] = t
                stock['after_maxdowntown'] = maxdowntown
                new_data[stocks] = stock
            except:
                pass
        del data
        gc.collect()
        return new_data
    def index_chuli(self,index_data):

        pindex_data = pd.DataFrame()
        pindex_data['close'] = list(index_data['close'])
        pindex_data['volume'] = list(index_data['volume'])
        #print (pindex_data.head(10))
        #获取20、个5个交易日的收益
        pindex_data['20_profit'] = pindex_data['close']/(pindex_data.shift(20)['close'])-1
        pindex_data['5_profit'] = pindex_data['close']/(pindex_data.shift(5)['close'])-1
        # 获取，当前价格和5日均价，20日均价之间的收益情况
        pindex_data_mean5 = pindex_data.rolling(5).mean()
        pindex_data_mean20 = pindex_data.rolling(20).mean()
        pindex_data['20_mean'] = (pindex_data['close'])/(pindex_data_mean20['close']) -1
        pindex_data['5_mean'] = (pindex_data['close'])/(pindex_data_mean5['close']) -1
        # 获取当前成交量与5日和20日均成交量之间的关系
        pindex_data['20_mean_vol'] = (pindex_data['volume'])/(pindex_data_mean20['volume']) -1
        pindex_data['5_mean_vol'] = (pindex_data['volume'])/(pindex_data_mean5['volume']) -1
        pindex_data['date'] = list(index_data['date'])
        #print (pindex_data.head(10))
        #print (list(pindex_data.columns))
        return pindex_data
    def factor_get(self):
        '''
        本函数的作用是为了，构建相应的基础因子模块。包括成交价成交量等
        本次试验，基础factors 为 开、收、高、低价格，成交量，5日均价，5日均成交量，5日内最高价
        '''
        stock_data = self.adaptability_compute()
        index_data = self.index_get()
        index_data = self.index_chuli(index_data = index_data)
        # 对指数进行处理，获取指数的当日的涨跌幅情况，
        # 前20个交易日的涨跌幅,前5个交易日的涨跌幅，当日涨跌幅
        # 今天较20日均值的涨跌幅，今日较5日均值的涨跌幅
        # 几日较20个交易日均值的成交量，今日缴5个个交易日的成家量
        index_data = index_data[['date','20_profit','20_mean','5_mean','20_mean_vol','5_mean_vol']]
        for stock in list(stock_data.keys()):
            stock_data[stock]['mean_five'] = (stock_data[stock][['close','low']].rolling(5).mean())['close']
            
            stock_data[stock]['highest_five'] = (stock_data[stock][['high','low']].rolling(5).max())['high']
            stock_data[stock]['lowest_five'] = (stock_data[stock][['close','low']].rolling(5).min())['low']
            stock_data[stock]['mean_20'] = (stock_data[stock][['close','low']].rolling(20).mean())['close']
            stock_data[stock]['highest_20'] = (stock_data[stock][['high','low']].rolling(20).max())['high']
            stock_data[stock]['lowest_20'] = (stock_data[stock][['close','low']].rolling(20).min())['low']
            stock_data[stock]['date'] = list(stock_data[stock].index)
            stock_data[stock] = pd.merge(stock_data[stock],index_data,on='date',how='inner')
        return stock_data
        
    #定义一个函数，按照时间将股票,和后续收益进行分割进行分割，并对有NAN数据进行剔除
    
    def time_adjust(self):
        all_data = self.factor_get()
        index_data = self.index_get()
        columns = list(all_data[self.stock_list[0]].columns)
        adjusted_factor = ['after_5_profit','after_maxdowntown']
        for i in adjusted_factor:
            columns.remove(i)
        columns.remove('code')
        #print (columns)
        time_list = list(index_data['date'])
        changdu =len(time_list)
        t_columns = copy.deepcopy(columns)
        t_columns.remove('date')
        #print (list((all_data['000333'])['date']))
        x_datas = [[] for i in range(len(time_list))]
        y_datas = [[] for i in range(len(time_list))]
        f = 0
        nan_value = [np.nan for i in t_columns]
        print (len(nan_value),t_columns)
        for stock in list(all_data.keys()):
            stockdata = all_data[stock]
            x_data = stockdata[columns]
            print (stock,f)
            f+=1
            y_data = stockdata[adjusted_factor+['date']]
            del stockdata
            for i in range(len(time_list)):
                now_time = time_list[i]
                x_stkdata = x_data[x_data['date'] ==now_time]
                y_stkdata= y_data[y_data['date'] == now_time]
                new_x = np.array(x_stkdata[t_columns])
                #print (x_stkdata[t_columns].head())
                #input()
                new_y = list(np.array(y_stkdata['after_5_profit']))
                #print (new_x,new_y,u'herere')
                if len(new_y)==0 or len(new_x)==0:
                    x_datas[i].append(nan_value)
                    y_datas[i].append(np.nan)
                
                else:
                    #print (len(new_x[0]),u'here2',new_y[0])
                    x_datas[i].append(list(new_x[0]))
                    y_datas[i].append(new_y[0])
                    #input()
            del x_data,y_data
            gc.collect()
        x_train_data = (x_datas[20:int(changdu*self.train_ratio)])
        y_train_data = y_datas[20:int(changdu*self.train_ratio)]
        x_test_data = x_datas[int(changdu*self.train_ratio):-5]
        y_test_data = y_datas[int(changdu*self.train_ratio):-5]
        return (x_train_data,y_train_data,x_test_data,y_test_data)
                
                
        
    #定义一个函数，用于进行train_data 与test_data，以及适应度数据选取
    def train_test(self):
        '''
        本代码的作用是为了区分，train,test ,x,y
        '''
        all_data = self.factor_get()
        columns = list(all_data[self.stock_list[0]].columns)
        
        adjusted_factor = ['after_5_profit','after_maxdowntown']
        for i in adjusted_factor:
            columns.remove(i)
        columns.remove('date')
        columns.remove('code')
        new_stock_data_train = OrderedDict()
        new_adjust_data_train = OrderedDict()
        new_stock_data_test = OrderedDict()
        new_adjust_data_test = OrderedDict()
        
        for stock in self.stock_list:
            changdu = len(all_data[stock]['open'])
            x_data = np.array(all_data[stock][columns])
            y_data = np.array(all_data[stock][adjusted_factor])
            new_stock_data_train[stock] = x_data[21:int(changdu*self.train_ratio),:]
            new_adjust_data_train[stock] = y_data[21:int(changdu*self.train_ratio),:]
            new_stock_data_test[stock] = x_data[int(changdu*self.train_ratio):,:-20]
            new_adjust_data_test[stock] =y_data[int(changdu*self.train_ratio):,:-20]
            #input()
        return (new_stock_data_train,new_adjust_data_train,new_stock_data_test,new_adjust_data_test)
#定以一个函数，用于对保存后读取的数据进行净化处理
#@numba.jit
def jinghua_data(x_train,y_train,x_test,y_test):
    x_trains = [[] for i in range(len(x_train))]
    y_trains = [[] for i in range(len(x_train))]
    x_tests  =  [[] for i in range(len(x_test))]
    y_tests  =  [[] for i in range(len(x_test))]
    (trains_time,trains_stock,f) = x_train.shape
    (test_time,test_stock,f) = x_test.shape
    for i in range(trains_time):
        for j in range(trains_stock):
            #print (x_train[i,j])
            if not kong_pan(x_train[i,j]) and abs(y_train[i][j])>=0:

                x_trains[i].append(list(x_train[i,j]))
                y_trains[i].append(y_train[i][j])
    '''
    for t in range(test_time):
        for k in range(test_stock):
            if not kong_pan(x_test[t,k]) or abs(y_test[t][k])>=0:
                x_tests[t].append(list(x_test[t,k]))
                y_tests[t].append(y_test[t][k])
    '''
    return (x_trains,y_trains,x_test,y_tests)
            
# 定义一个函数，用于进行遗传进化    
if __name__ == '__main__':
    '''
    stocks = list(pd.read_csv('C:\\Users\\94006\\Desktop\\HS300.csv')['HS300'])[0:150]
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
    print (u'ppppppppppppppppp')
    x_train_HS300 = 'D:\\HS300_x_train_datas150.npy'
    y_train_HS300 = 'D:\\HS300_y_train_datas150.npy'
    x_test_HS300 = 'D:\\HS300_x_test_datas150.npy'
    y_test_HS300 = 'D:\\HS300_y_test_datas150.npy'
    np.save(x_train_HS300,np.array(x_train))
    np.save(y_train_HS300,np.array(y_train))
    np.save(x_test_HS300,np.array(x_test))
    np.save(y_test_HS300,np.array(y_test))
    '''
   
    x_train = np.array(np.load('D:\\HS300_x_train_datas150.npy'))
    y_train = np.array(np.load('D:\\HS300_y_train_datas150.npy'))
    x_test = np.array(np.load('D:\\HS300_x_test_datas150.npy'))
    y_test =np.array( np.load('D:\\HS300_y_test_datas150.npy'))
    print (x_train[:,0:2])
    input()
    
    (a,b,c,d) = jinghua_data(x_train,y_train,x_test,y_test)
 
    print (u'数据准备完成，进入进化')
    est_gp = SymbolicRegressor(population_size=200,
                           generations=8, stopping_criteria=10000,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0,
                           metric= 'stock_dedicated',
                           n_jobs=2)# 构建一个遗传进化的类 
    print (u'类构件完成')
    input()                        
    x_trains = a
    #print (a)
    y_trains = b
    
    est_gp.fit(x_trains, y_trains)   
