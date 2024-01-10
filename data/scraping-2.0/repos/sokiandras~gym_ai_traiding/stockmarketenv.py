import random
import gym
import numpy as np
import yfinance as yf
import datetime
import os
import pandas as pd
import openai
from newsapi import NewsApiClient


class StockMarketEnv(gym.Env):
   #metadata = {'render.modes': ['human']}

   def __init__(self, symbol, start_date, end_date, balance, log):
       self.stock_symbol = symbol
       self.start_date = start_date
       self.end_date = end_date
       self.log = log

       self.data = yf.download(self.stock_symbol, self.start_date, self.end_date, interval="1h")  #itt tölti le az adatokat
       self.stepnumber = len(self.data)
       self.current_step = None  # amikor először lefut, akkor még nincs step

       # kezdő beállítások
       self.start_balance = balance  # kezdő egyenleg
       self.known_data_number = 60  # amennyi árat ismer maga előtt
       self.current_step = 0
       self.current_info = 0

       #action space beállításai
       low_a = np.array([0,0])
       high_a = np.array([3,1])
       self.action_space = gym.spaces.Box(low_a, high_a, dtype = np.float32)   # első dimenzió: 0-3 között bármi = 0-1: elad, 1-2: tart, 2-3: vesz;  második dimenzió: 0-1 között bármi = mekkora hányadát költi a pénzének a műveletre.
       # self.action_space = gym.spaces.Discrete(3)  # legegyszerűbb módszer, vagy elad, vagy vesz, vagy tart


       #observation space beállításai
       low_o = 0
       high_o = 1
       shape = (6, self.known_data_number)  # így az obs_space úgy fog kinézni, hogy az egyik dimenzió 6 (mivel 6 adatot kap meg a yf által letöltött adatokból - 1 időpontra 6 adat van (high, low, stb.), a másik dimenzió pedig a known_data_number (vagyis azok a sorok amikre visszalát)
       self.observation_space = gym.spaces.Box(low_o, high_o, shape, dtype = np.float32)
       #self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,))



   def get_observation(self):
       # annyi adatsort ismer maga mögött, mint amennyi a a known_data_number

       # normálásokhoz a maximumok
       max_open = self.data['Open'].max()
       max_high = self.data['High'].max()
       max_low = self.data['Low'].max()
       max_close = self.data['Close'].max()
       max_adjclose = self.data['Adj Close'].max()
       max_volume = self.data['Volume'].max()

       max_balance = self.start_balance*100 # ha olyan jó lesz, hogy 1000x-esére növelné a pénzt, akkor ezt át kell írni

       frame = np.array([
           self.data.iloc[self.current_step - self.known_data_number: self.current_step]['Open'].values/max_open,  #open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
           self.data.iloc[self.current_step - self.known_data_number: self.current_step]['High'].values/max_high,  # open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
           self.data.iloc[self.current_step - self.known_data_number: self.current_step]['Low'].values/max_low,  # open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
           self.data.iloc[self.current_step - self.known_data_number: self.current_step]['Close'].values/max_close,  # open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
           self.data.iloc[self.current_step - self.known_data_number: self.current_step]['Adj Close'].values/max_adjclose,  # open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
           self.data.iloc[self.current_step - self.known_data_number: self.current_step]['Volume'].values/max_volume,  # open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
       ]) # known_data_number oszlopa és 6 sora lesz: 1. sor: open, 2.: high, 3.: close, .... és az oszlopok a dátumok száma

       # myFavouriteVal = [[self.balance / max_balance]]

       # observation = np.append(frame, myFavouriteVal, axis=1)

       observation = frame

       return observation



   def reset(self):
       self.current_step = self.known_data_number+1
       self.balance = self.start_balance
       self.net_worth = self.start_balance
       #self.prev_balance = self.start_balance
       self.total_reward = 0
       self.total_profit = 0

       # arra vonatkozik, hogy hány részvény db-ot vettünk/eladtunk összesen (ha 100 db apple részvényt vettünk, akkor itt 100 db fog látszódni)
       self.total_bought_open = 0
       self.total_bought_closed = 0
       self.total_shorted_open = 0
       self.total_shorted_closed = 0
       self.sales_log = []

        # a jövőben arra fog vonatkozni, hogy milyen fajta részvénnyel történt. pl. ha apple-ből veszünk 100 db-ot, akkor itt 1 db apple jelenik meg
       self.share_bought_open = 0
       self.share_bought_closed = 0
       self.share_shorted_open = 0
       self.share_shorted_closed = 0

       return self.get_observation()



   def take_action(self, action):
       action_type = action[0]
       balance_usage_ratio = action[1]
       sell_ratio = action[1]

       #self.current_info = ""

       open_price = self.data.iloc[self.current_step]['Open']
       close_price = self.data.iloc[self.current_step]['Close']

       current_price = random.uniform(open_price, close_price) # az adott lépésnél az Open és a Close ár között választ egy véletlen árat
       self.prev_net_worth = self.net_worth


       if action_type <= 1: # vásárlás  (annyi darabot, ahányad részét a balance_usage_ratio enged)
            type = 'Buy'
            if (balance_usage_ratio == 0):
                possibility = 0
                possibility_reason = 'Buy, but balance_usage_ration = 0'
                self.current_info = {
                    'step: ': self.current_step,
                    'possibility: ': possibility,
                    'possibility reason: ': possibility_reason,
                }
            else:
                possible_number_of_boughts = (self.balance * balance_usage_ratio / current_price) // 1  # ennyi darabot lehet max venni (a //1 az alsó egészrész
                if (possible_number_of_boughts > 0):
                        possibility = 1
                        self.total_bought_open = self.total_bought_open + possible_number_of_boughts
                        additional_cost = possible_number_of_boughts * current_price
                        self.balance = self.balance - additional_cost
                        self.current_info = {
                            'step: ': self.current_step,
                            'type: ': type,
                            'possibility: ': possibility,
                            'bought pieces: ': possible_number_of_boughts,
                            'current price ': current_price,
                            'cost: ': additional_cost,
                        }
                else:
                    possibility = 0
                    possibility_reason = 'Buy, but possible number of boughts = 0'
                    self.current_info = {
                        'step: ': self.current_step,
                        'type: ': type,
                        'possibility: ': possibility,
                        'possibility reason: ': possibility_reason,
                    }


       if action_type > 1 and action_type <= 2 :  # hold
           type = 'hold'
           self.current_info = {
               'step: ': self.current_step,
               'type: ': type,
           }


       if action_type > 2 and action_type <= 3: # elad (annyi darabot, ahányad részt megad a sell_ratio)
           if (sell_ratio == 0):
               possibility = 0
               possibility_reason = 'Sell, but sell_ration = 0'
               self.current_info = {
                   'step: ': self.current_step,
                   'possibility: ': possibility,
                   'possibility reason: ': possibility_reason,
               }
           else:
               possible_number_of_sells = (self.total_bought_open * sell_ratio) //1  # ennyi darabot ad el
               type = 'sell'
               if (possible_number_of_sells > 0):
                   possibility = 1
                   self.total_bought_closed = self.total_bought_closed + possible_number_of_sells
                   self.total_bought_open = self.total_bought_open - possible_number_of_sells
                   additional_income = current_price * possible_number_of_sells
                   self.balance = self.balance + additional_income
                   self.current_info = {
                       'step: ': self.current_step,
                       'type: ': type,
                       'possibility: ': possibility,
                       'sold pieces: ': possible_number_of_sells,
                       'current price ': current_price,
                       'income: ': additional_income,
                   }
                   #self.sales_log.append(self.current_info)
               else:
                   possibility = 0
                   possibility_reason = 'Sell, but possible number of sells = 0'
                   self.current_info = {
                       'step: ': self.current_step,
                       'type: ': type,
                       'possibility: ': possibility,
                       'possibility reason: ': possibility_reason,
                   }
                   #self.sales_log.append(self.current_info)



       if (action[0] == 0 and action[1] == 0):
           possibility =0
           possibility_reason = 'action = [0 0]'
           self.current_info = {
               'step: ': self.current_step,
               'possibility: ' : possibility,
               'possibility reason: ': possibility_reason
           }


       self.current_info['total open shares: '] = self.total_bought_open
       self.current_info['total sold shares so far: '] = self.total_bought_closed
       self.current_info['current balance: '] = self.balance
       self.net_worth = self.balance + self.total_bought_open * current_price
       self.current_info['current net worth: '] = self.net_worth
       self.current_info['action: '] = action

       if self.current_info:
            self.sales_log.append(self.current_info)

       return self.current_info




   def step(self, action):

       time = self.data.index[self.current_step]
       if(self.log):
            print(time)  #csak azért, hogy külön sorba írja

       current_info = self.take_action(action)
       if(self.log):
            print(current_info)  # minden step current infoja (kivéve time és profit)

       reward = self.net_worth - self.prev_net_worth
       if(self.log):
            print('Reward (profit of the step): ', reward)  # csak azért van így, hogy külön sorba írja ki, így látványosabb legyen a profit

       self.current_info['Reward = Profit: '] = reward
       self.current_info['Time: '] = time


       self.current_step = self.current_step + 1

       done = False
       if (self.current_step == len(self.data)):
           done = True
           total_profit = self.net_worth - self.start_balance
           print('comment from env: Last step on dataline')
           total_profit_to_log = {'Total profit: ', total_profit}
           print('comment from env: Total profit: ', total_profit)
           self.sales_log.append(total_profit_to_log)
           print('comment from env: data length: ', len(self.data))

       observations = self.get_observation()

       return observations, reward, done, current_info


   def close(self):
       print('Close')


   def log_writer(self):
       timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
       log_filename = f'stock_trading_log_{timestamp}.txt'
       log_folder_path = "D:\Egyetem\Önlab\sajat1\logs"
       if not os.path.exists(log_folder_path):
           os.makedirs(log_folder_path)

       logfile = os.path.join(log_folder_path,log_filename)
       with open(logfile,"x") as file:
           for i in self.sales_log:
               file.write('%s\n' % i)

