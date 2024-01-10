import random
import gym
import numpy as np
import yfinance as yf
import datetime
import os
import pandas as pd
from analyze_news import Analyze_news
import openai



class Env_with_news(gym.Env):

    def __init__(self, symbol, start_date, end_date, balance, log, data_interval_like_1h):
        self.stock_symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.log = log
        self.data_interval = data_interval_like_1h  #így kell kinézni "1h" (STRING!!!)

        # self.data = yf.download(self.stock_symbol, self.start_date, self.end_date, interval=self.data_interval)  # itt tölti le az adatokat
        #self.stepnumber = len(self.data)
        self.current_step = None  # amikor először lefut, akkor még nincs step

        # kezdő beállítások
        self.start_balance = balance  # kezdő egyenleg
        self.known_data_number = 3  # amennyi árat ismer maga előtt
        self.current_step = 0
        self.current_info = 0


        self.news_scores = []
        self.news_scores_with_index = pd.DataFrame
        self.reducated_news_scores = pd.DataFrame

        # action space beállításai
        low_a = np.array([0, 0])
        high_a = np.array([3, 1])
        self.action_space = gym.spaces.Box(low_a, high_a, dtype=np.float32)  # első dimenzió: 0-3 között bármi = 0-1: elad, 1-2: tart, 2-3: vesz;  második dimenzió: 0-1 között bármi = mekkora hányadát költi a pénzének a műveletre.
        # self.action_space = gym.spaces.Discrete(3)  # legegyszerűbb módszer, vagy elad, vagy vesz, vagy tart

        # observation space beállításai
        low_o = 0
        high_o = 1
        shape = (7, self.known_data_number)  # így az obs_space úgy fog kinézni, hogy az egyik dimenzió 6 (mivel 6 adatot kap meg a yf által letöltött adatokból - 1 időpontra 6 adat van (high, low, stb.), a másik dimenzió pedig a known_data_number (vagyis azok a sorok amikre visszalát)
        self.observation_space = gym.spaces.Box(low_o, high_o, shape, dtype=np.float32)
        # self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,))

        # a NewsAPI miatt nem lehet 1 hónapnál régebbre visszamenni
        current_date = datetime.datetime.now()
        #if (current_date - start_date) > 30:
            #raise ValueError("\nStart date is older than one month from the current date, so NewsAPI will not be able to get news")

        self.data_maker()  # itt tölti le az adatokat és egyesíti a hírekkel
        self.stepnumber = len(self.data)





    def news_analysis_in_given_interval(self):

        start_date = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        current_date = start_date

        interval_in_int = int(self.data_interval[:-1])

        if self.data_interval.endswith('h'):
            delta = datetime.timedelta(hours=interval_in_int)
        elif self.data_interval.endswith('d'):
            delta = datetime.timedelta(days=interval_in_int)
        else:
            if self.log <= 3:
                print("\nThe interval is not correctly given (message from news_analysis_in_given_interval()")
            return -1


        while current_date <= end_date:
            current_hour = current_date.strftime("%H")
            current_date_string = current_date.strftime("%Y-%m-%d")

            news_analyzer = Analyze_news(self.stock_symbol, current_date_string, hour=int(current_hour), log =self.log)
            try:
                average_score = news_analyzer.analyze()
            except openai.error.InvalidRequestError as e:
                if self.log <= 3:
                    print(f"\nError occurred while analyzing news: {str(e)} news details: {current_date}, hour: {current_hour}")
                    print("\nSkipping this news and continuing with the next one.")
                current_date += delta
                average_score = -1

            if self.log <= 3:
                print(f"\nAnalysis for {current_date_string} - hour: {current_hour} - result score: {average_score} (message from news_analysis_in_given_interval())")


            if average_score != -1:
                self.news_scores.append(average_score)
            else:
                self.news_scores.append(5)

            current_date += delta

        print(f"\n\n\nNews scores: {self.news_scores} (message from news_analysis_in_given_interval())")




    def set_datetime_index_for_data(self):
        string_index = self.data.index.strftime('%Y-%m-%d %H')
        datetime_index = pd.to_datetime(string_index)
        self.data.index = datetime_index

        if self.log <= 3:
            print('\ndata:\n')
            print(self.data)




    def set_datetime_index_for_news_scores(self):
        news_scores_with_index = pd.Series(self.news_scores)

        datetime_strings = []
        current_date = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        interval_in_int = int(self.data_interval[:-1])

        if self.data_interval.endswith('h'):
            delta = datetime.timedelta(hours=interval_in_int)
        elif self.data_interval.endswith('d'):
            delta = datetime.timedelta(days=interval_in_int)

        while len(datetime_strings) < len(news_scores_with_index):
            current_hour = current_date.strftime("%H")
            current_date_string = current_date.strftime("%Y-%m-%d")
            datetime_strings.append(f"{current_date_string} {current_hour}")
            current_date += delta

        index_in_datetime = pd.to_datetime(datetime_strings)
        news_scores_with_index.index = index_in_datetime

        self.news_scores_with_index = news_scores_with_index

        if self.log <= 3:
            print(f"\nNews score in indexed series: (message from set_datetime_index_for_news_scores())\n ", self.news_scores_with_index)




    def give_as_many_news_scores_as_dataline(self):
        self.set_datetime_index_for_data()
        self.set_datetime_index_for_news_scores()

        new_scores = []
        temp_scores = []

        self.data.index = self.data.index.strftime('%Y-%m-%d %H')
        self.news_scores_with_index.index = self.news_scores_with_index.index.strftime('%Y-%m-%d %H')


        for index in self.news_scores_with_index.index:
            if index in self.data.index:
                if temp_scores:  # if temp_scores is not empty
                    temp_scores.append(self.news_scores_with_index.loc[index])
                    average_score = sum(temp_scores) / len(temp_scores)
                    score = average_score
                    temp_scores = []  # reset temp_scores
                else:
                    score = self.news_scores_with_index.loc[index]

                new_scores.append(score)
            else:
                temp_scores.append(self.news_scores_with_index.loc[index])


        self.reducated_news_scores = pd.Series(new_scores, index = self.data.index)

        if self.log <= 3:
            print("\n\nReducated news scores list: (message from give_as_many_news_scores_as_dataline())", self.reducated_news_scores)





    def data_maker(self):
        self.data = yf.download(self.stock_symbol, self.start_date, self.end_date, interval=self.data_interval)
        self.news_analysis_in_given_interval()
        self.give_as_many_news_scores_as_dataline()
        self.data['News scores'] = self.reducated_news_scores

        print('\n\n data: ')
        print(self.data.head(10))







    def get_observation(self):
        # annyi adatsort ismer maga mögött, mint amennyi a a known_data_number

        # normálásokhoz a maximumok
        max_open = self.data['Open'].max()
        max_high = self.data['High'].max()
        max_low = self.data['Low'].max()
        max_close = self.data['Close'].max()
        max_adjclose = self.data['Adj Close'].max()
        max_volume = self.data['Volume'].max()
        max_news_score = 10

        max_balance = self.start_balance * 100  # ha olyan jó lesz, hogy 1000x-esére növelné a pénzt, akkor ezt át kell írni


        frame = np.array([
            self.data.iloc[self.current_step - self.known_data_number: self.current_step]['Open'].values / max_open,
            # open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
            self.data.iloc[self.current_step - self.known_data_number: self.current_step]['High'].values / max_high,
            # open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
            self.data.iloc[self.current_step - self.known_data_number: self.current_step]['Low'].values / max_low,
            # open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
            self.data.iloc[self.current_step - self.known_data_number: self.current_step]['Close'].values / max_close,
            # open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
            self.data.iloc[self.current_step - self.known_data_number: self.current_step]['Adj Close'].values / max_adjclose,
            # open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
            self.data.iloc[self.current_step - self.known_data_number: self.current_step]['Volume'].values / max_volume,
            # open lenormálva (attól amennyitől ismeri az adatokat a jelenlegi lépésig)
            self.data.iloc[self.current_step - self.known_data_number: self.current_step]['News scores'].values / max_news_score
        ])  # known_data_number oszlopa és 6 sora lesz: 1. sor: open, 2.: high, 3.: close, .... és az oszlopok a dátumok száma

        # myFavouriteVal = [[self.balance / max_balance]]

        # observation = np.append(frame, myFavouriteVal, axis=1)

        observation = frame

        return observation






    def reset(self):
        self.current_step = self.known_data_number + 1
        self.balance = self.start_balance
        self.net_worth = self.start_balance
        # self.prev_balance = self.start_balance
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

        # self.current_info = ""

        open_price = self.data.iloc[self.current_step]['Open']
        close_price = self.data.iloc[self.current_step]['Close']

        current_price = random.uniform(open_price, close_price)  # az adott lépésnél az Open és a Close ár között választ egy véletlen árat
        self.prev_net_worth = self.net_worth

        if action_type <= 1:  # vásárlás  (annyi darabot, ahányad részét a balance_usage_ratio enged)
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

        if action_type > 1 and action_type <= 2:  # hold
            type = 'hold'
            self.current_info = {
                'step: ': self.current_step,
                'type: ': type,
            }

        if action_type > 2 and action_type <= 3:  # elad (annyi darabot, ahányad részt megad a sell_ratio)
            if (sell_ratio == 0):
                possibility = 0
                possibility_reason = 'Sell, but sell_ration = 0'
                self.current_info = {
                    'step: ': self.current_step,
                    'possibility: ': possibility,
                    'possibility reason: ': possibility_reason,
                }
            else:
                possible_number_of_sells = (self.total_bought_open * sell_ratio) // 1  # ennyi darabot ad el
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
                    # self.sales_log.append(self.current_info)
                else:
                    possibility = 0
                    possibility_reason = 'Sell, but possible number of sells = 0'
                    self.current_info = {
                        'step: ': self.current_step,
                        'type: ': type,
                        'possibility: ': possibility,
                        'possibility reason: ': possibility_reason,
                    }
                    # self.sales_log.append(self.current_info)

        if (action[0] == 0 and action[1] == 0):
            possibility = 0
            possibility_reason = 'action = [0 0]'
            self.current_info = {
                'step: ': self.current_step,
                'possibility: ': possibility,
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
        if (self.log):
            print(time)  # csak azért, hogy külön sorba írja

        current_info = self.take_action(action)
        if (self.log):
            print(current_info)  # minden step current infoja (kivéve time és profit)

        reward = self.net_worth - self.prev_net_worth
        if (self.log):
            print('Reward (profit of the step): ',
                  reward)  # csak azért van így, hogy külön sorba írja ki, így látványosabb legyen a profit

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

        logfile = os.path.join(log_folder_path, log_filename)
        with open(logfile, "x") as file:
            for i in self.sales_log:
                file.write('%s\n' % i)




# test_env = Env_with_news('AAPL','2023-11-30', '2023-12-02',100000,2,'1h')
# test_env.news_analysis_in_given_interval()
# test_env.give_as_many_news_scores_as_dataline()

#test_env = Env_with_news('AAPL','2023-11-30', '2023-12-02',100000,2,'1h')

