#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:30:55 2022

@author: litingtang
"""
import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
import time
import requests
import json
from random import seed
import random
import matplotlib.pyplot as plt

random.seed(10)

#https://financialmodelingprep.com/api/v3/market-capitalization/AAPL?apikey=YOUR_API_KEY
api_key = "48ac1e227536f2b7eb100c88841ff2cd"
api_url = "https://financialmodelingprep.com/api/v3/market-capitalization/"



def get_marketcap(symbol):
    '''
    
    Parameters
    ----------
    symbol : string
        Ticker of the stocks.

    Returns
    -------
    integer
        market capitalization of stock.

    '''
    url = api_url+symbol+'?apikey='+api_key
    response = requests.get(url)
    data = json.loads(response.text)
    
    return data[0]['marketCap']

def get_price(ticker_list, period='1mo'):
    '''
    
    Parameters
    ----------
    ticker_list : list
        list of the ticker.
        
    period : string
        1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        
    Returns
    -------
    price : dataframe
        historical stock price.

    '''
    ticker = yf.Tickers(ticker_list)
    price = ticker.history(period=period)['Close'].fillna(method='ffill')
    
    return price 

def generate_random_deposit(num_of_day, start, stop):
    '''
    
    Parameters
    ----------
    num_of_day : integer
        number of random deposits to be generated.
    
    start : intger / float
        lower bar of the random generated numbers.
    
    stop : integer / float
        upper bar of the random generated numbers.
        
    Returns
    -------
    random_deposit : numpy array
        one array of the random number.
    
    '''
    random_deposit = random.sample(range(start, stop), num_of_day)
    
    return random_deposit


def cal_ret(df):
    return np.log(df) - np.log(df.shift(1))
    # return df.shift(1)/df-1

def stop_loss_standard(weighted_std):
    
    if weighted_std > 0.003:
        sl = -0.05
    else:
        sl = -0.1
    return sl

#%%
# if __name__ == '__main__':
    
tech_giant = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'BABA', 'CRM', 
              'AMD', 'INTC', 'PYPL', 'ATVI', 'EA', 'TTD', 'MTCH', 'ZG', 'YELP']

#historical price of components
tech_price = get_price(tech_giant, '1Y')
tech_ret = tech_price / tech_price.iloc[0]
#current marketcap of components
tech_mc = pd.DataFrame(tech_giant, columns=['Ticker'])
tech_mc['marketCap'] = tech_mc['Ticker'].apply(get_marketcap)
tech_mc['marketWeight'] = tech_mc['marketCap'] / sum(tech_mc['marketCap'])

#%%
#calculate the portfolio weights 
views = [0] * len(tech_giant)
views= [-0.07, -0.05, -0.03, -0.01, 0, 0, 0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
tech_mc['portfolio_weight'] = tech_mc['marketWeight'] + views

#deposit simulation/ portfolio simulation
random_deposit = generate_random_deposit(len(tech_price), 10, 10000)
random_deposit_arr = np.array(random_deposit)[None].T
p_weight_arr = tech_mc['portfolio_weight'].values[None]
net_value_each = pd.DataFrame(random_deposit_arr*p_weight_arr, columns=tech_giant)
net_value_each = net_value_each.set_index(tech_price.index)

#cumualtive net value of the portfolio
portfolio_nv = pd.DataFrame(net_value_each.sum(axis=1)) #sum up net value of position horizontally and get portfolio net value
portfolio_nv = portfolio_nv.set_index(net_value_each.index) 
cumu_nv = pd.DataFrame(portfolio_nv.cumsum(axis=0)) #calculate cumulative summation of portfolio from the very begining of the dataset

#%%
#calculate B book's average cost of each position on each date
shares_each = net_value_each / tech_price
cumu_shares_each = shares_each.cumsum(axis=0)
position_nv = net_value_each.cumsum(axis=0)
avg_cost = position_nv / cumu_shares_each

#calculate B book pnl
cumu_pnl = pd.DataFrame((tech_price - avg_cost) * cumu_shares_each)
b_pnl_by_pos = cumu_pnl / position_nv
cumu_b_pnl = pd.DataFrame(-(cumu_pnl.sum(axis=1)))
cumu_b_pnl_per = pd.DataFrame(cumu_b_pnl.div(cumu_nv, axis=0))

#set stop loss standard of each position
tech_ret = cal_ret(tech_price)
tech_std = tech_ret.std()
tech_weighted_std = tech_std.mul(tech_mc['portfolio_weight'].values, axis=0)
sl_standard = pd.DataFrame(tech_weighted_std).reset_index()
sl_standard.columns = ['Ticker', 'weighted_std']
sl_standard['stop_loss'] = sl_standard['weighted_std'].apply(stop_loss_standard)
sl_standard = sl_standard.set_index(['Ticker'])
sl_standard['stop_loss'] = sl_standard['stop_loss'] + 0.02

#%%

#calculate A book pnl
ret_vs_avg = pd.DataFrame(avg_cost / tech_price - 1)
cumu_a_book = pd.DataFrame(np.zeros(ret_vs_avg.shape), columns=tech_giant)
cumu_a_book = cumu_a_book.set_index(ret_vs_avg.index)
#if position ret < position sl standard: a book = cumu_shares_each * tech_price

a_flag = [False]*len(tech_giant)
cumu_pos_days = 2
cumu_a_book = cumu_a_book.reset_index()
ret_vs_avg = ret_vs_avg.reset_index()

for i in range(cumu_pos_days, cumu_a_book.shape[0]):
    for j, symbol in enumerate(tech_giant):
            
        if ret_vs_avg[symbol][i] <= sl_standard['stop_loss'][symbol] or a_flag[j]:
            cumu_a_book[symbol][i] = tech_price[symbol][i]*cumu_shares_each[symbol][i]
            a_flag[j] = True
        
        if sum(ret_vs_avg[symbol][i-cumu_pos_days+1:i+1]>0) == cumu_pos_days:
            a_flag[j] = False
    
a_pnl = np.zeros(cumu_a_book.shape)
cumu_a_book_arr = cumu_a_book.values[:, 1:]
a_pnl = np.zeros(cumu_a_book_arr.shape)
tech_price_arr = tech_price.values[:,]
shares_each_arr = shares_each.values[:,]
for i in range(1, cumu_a_book_arr.shape[0]):
    for j in range(cumu_a_book_arr.shape[1]):
        if cumu_a_book_arr[i][j]!=0 and cumu_a_book_arr[i-1][j]==0:
            a_pnl[i][j] = 0
        elif cumu_a_book_arr[i][j]!=0:
            a_pnl[i][j] = cumu_a_book_arr[i][j]-cumu_a_book_arr[i-1][j]
            a_pnl[i][j] = a_pnl[i][j]-(tech_price_arr[i][j]*shares_each_arr[i][j])
        else:
            a_pnl[i][j] = 0

a_pnl_df = pd.DataFrame(a_pnl, columns=tech_giant)
a_pnl_df.insert(0, 'Date', cumu_a_book['Date'])
a_pnl_df.set_index(tech_price.index)
    
a_pnl_portfolio_df = a_pnl_df.iloc[:, 1:].sum(1)
cumu_pnl_portfolio_df = a_pnl_portfolio_df.cumsum(axis=0)
cumu_pnl_portfolio_per = cumu_pnl_portfolio_df.values[None].T / cumu_nv.values
cumu_pnl_portfolio_per_df = pd.DataFrame(cumu_pnl_portfolio_per)
cumu_pnl_portfolio_per_df = cumu_pnl_portfolio_per_df.set_index(cumu_b_pnl_per.index)
pnl = cumu_b_pnl.values + cumu_pnl_portfolio_df.values[None].T
pnl_per = pd.DataFrame(pnl /cumu_nv.values)
pnl_per = pnl_per.set_index(cumu_b_pnl_per.index)

  

#%%
#import seaborn
plt.figure(figsize=(10, 8)) 

plt.plot(cumu_b_pnl_per, label='b_pnl')
plt.plot(pnl_per, label='a+b')
plt.plot(cumu_pnl_portfolio_per_df, label='a_pnl')

plt.ylim(-0.4, 0.4)
plt.legend()
plt.grid()
plt.show()    

# #%%
# plt.figure(figsize=(10, 8))
# plt.plot(cumu_nv)
    
#%%
plt.figure(figsize=(10, 8))
# plt.pie(tech_mc['portfolio_weight'],
#         labels=tech_mc['Ticker'], 
#         )

plt.pie(tech_mc['marketWeight'], 
        labels=tech_mc['Ticker'])

#%%

    
    
    
    