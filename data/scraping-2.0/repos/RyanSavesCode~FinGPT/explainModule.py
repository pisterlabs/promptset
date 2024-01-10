#%matplotlib inline
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
import openai

check_and_make_directories([TRAINED_MODEL_DIR])

import itertools

# Import ChatGPT Function
import openai
# Function to prompt ChatGPT with FinGPT prompt.
def explain(indicators, averages, values, stocks, results):
    #Access to ChatGPT requires an API Key. The Key below is limited in its use and will eventually be inactive.
    openai.api_key = "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"
    index_key = 0
    index_ind = 0
    starting_values = [0]
    prompt = "We trained a model with a2c DRL agent on the following tickers "
    prompt += str(list(stocks))
    
    #Format prompt
    for item in stocks.keys():
        for ind in indicators:
            prompt += ". " + str(item) + "'s average " + ind + " is " + str(averages[index_key][ind])
        #prompt += ". Today " + str(item) + "'s  vix is " + str(values['vix']) + ". Today " + str(item) + "'s turbulence is " + str(values['turbulence'])
        for ind in indicators:
            prompt += ". Today " + str(item) + "'s " + ind + " is " + str(values[index_key][ind])
        index_key+=1
    prompt += " I asked FinRL to suggest to either buy, sell, or hold the stock. Concisely explain why today's feature values would cause FinRL to suggest to \n"
    
    tickers = list(stocks.keys())
    result_values = []
    for item in stocks.keys():
        result_values.append(results[str(item)][0])
    start_values = list(stocks.values())
    for index in range(0,len(result_values)):
        if int(result_values[index]) < int(start_values[index]):
            prompt += "sell " + str(int(start_values[index]) - int(result_values[index])) + " shares of " + tickers[index]+ ".\n"
        elif int(result_values[index]) > int(start_values[index]):
            prompt += "buy " + str(int(result_values[index]) - int(start_values[index])) + " shares of " + tickers[index]+ ".\n"
        elif int(result_values[index]) == int(start_values[index]):
            prompt += "hold " + str(results[index]) + "shares of " + tickers[index]+ ".\n"
    
    #Refinement prompt additions
    prompt += "Don't talk about how little you know. Disregard features that would not have impact." 
    
    #ChatGPT model options
    model = "text-davinci-003"
    response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=500)
    generated_text = "\n\nExplanation:\n" + response.choices[0].text
    
    print(generated_text)
    return generated_text
    

# Function to prompt ChatGPT to elaborate on indicators.
def elaborate(last_entry):
        openai.api_key = "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"
        prompt = "Elaborate on what each feature means in the last explanation you gave: " 
        model = "text-davinci-003"
        response = openai.Completion.create(engine=model, prompt=prompt+last_entry, max_tokens=500)
        generated_text = "\n\nElaboration:\n" + response.choices[0].text
        print(generated_text)
        return generated_text