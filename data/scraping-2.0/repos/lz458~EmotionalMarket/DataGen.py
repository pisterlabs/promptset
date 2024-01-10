#1. Import requirements
import snscrape.modules.twitter as sntwitter
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import datetime
import numpy as np
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification
import requests 
import os
from google.cloud import storage
import yfinance as yf
from multiprocessing import Pool
import multiprocessing as mp
import time 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import precision_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import schedule
from io import BytesIO
import pymysql
import mysql.connector
from multiprocessing import Pool
import multiprocessing

import openai






#2. Process Market information 

openai.api_key = "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"


def GPT_analysis(sentence):
  response = openai.Completion.create(
  model="text-ada-001",
  prompt=f"Decide whether a newsheadline's sentiment is positive, neutral, or negative.\n\nnewsheadline: \"{sentence}\"\nSentiment:",
  temperature=0,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0.5,
  presence_penalty=0
)
  output = response['choices'][0]['text']
  print(output)
  return output
  



def Get_FT_Market():
 
        today = datetime.datetime.today()
        today = today.strftime("%Y-%m-%d") 
        finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        nlp = pipeline('sentiment-analysis', model=finbert, tokenizer=tokenizer)
        sectionList =['world', 
                    'global-economy', 
                    'world-uk', 
                    'us', 
                    'uk-business-economy', 
                    'uk-politics-policy', 
                    'uk-companies', 
                    'us-economy',
                    'us-companies',
                    'us-politics-policy'
                    'equities', #Can change for specific asset class in the future 
                    'us-equities',
                    'technology-sector'#Can change for specific sectors class in the future
                    ]


        Partlist = ["_Title", "_Subtitle",'_Date']
        finBertList = ["_finBERTLabel", "_finBERTScore"]
        variableList =[]


        for i in range(len(sectionList)):
            sectionname = sectionList[i]
            for j in range(len(Partlist)):
                partname = Partlist[j]
                new_varname = sectionname + partname
                vars()[new_varname] = []
                variableList.append(vars()[new_varname])
                for k in range(len(finBertList)):
                    finbertname = finBertList[k]
                    new_varname = sectionname + partname + finbertname
                    vars()[new_varname] = []
                    variableList.append(vars()[new_varname])




        for section in tqdm(sectionList): 
            print(f'{section} started')

            for pages in tqdm(range(1,201)):
                url="https://www.ft.com/{sections}?page={page}".format(page = pages, sections = section)
                result=requests.get(url)
                reshult=result.content
                soup=BeautifulSoup(reshult, "lxml")
                
                soup = soup.find("div", {'class': 'js-track-scroll-event'})

                if soup == None: 
                    continue
                else:
                        
                    for Card in soup.findAll("li",{"class":"o-teaser-collection__item o-grid-row"}):
                        if Card == None: 
                            continue
                        else:
                                
                            title = Card.find("div",{"class":"o-teaser__heading"})
                            if title == None:
                                vars()[section+'_Title'].append('none')
                                vars()[section+'_Title'+'_finBERTLabel'].append(0)
                                vars()[section+'_Title'+'_finBERTScore'].append(0)
                            else:
                                titles=title.text
                                vars()[section+'_Title'].append(titles)
                                vars()[section+'_Title'] = [str(di) if di is None else di for di in vars()[section+'_Title']]
                                finBERTresults = nlp(titles)
                                for items in finBERTresults:#convert into dictionary 
                                    vars()[section+'_Title'+'_finBERTLabel'].append(items['label'])
                                    vars()[section+'_Title'+'_finBERTScore'].append(items['score'])



                            Subtitle=Card.find("a",{"class":"js-teaser-standfirst-link"})
                            if Subtitle == None:
                                vars()[section+'_Subtitle'].append('none')
                                vars()[section+'_Subtitle'+'_finBERTLabel'].append(0)
                                vars()[section+'_Subtitle'+'_finBERTScore'].append(0)
                            else:
                                Subtitles=Subtitle.text
                                vars()[section+'_Subtitle'].append(Subtitles)
                                vars()[section+'_Subtitle'] = [str(di) if di is None else di for di in vars()[section+'_Subtitle']]
                                finBERTresults = nlp(Subtitles)
                            
                                for items in finBERTresults:
                                    vars()[section+'_Subtitle'+'_finBERTLabel'].append(items['label'])
                                    vars()[section+'_Subtitle'+'_finBERTScore'].append(items['score'])


                            Date=Card.find("time",{"class":"o-date"})
                            if Date == None:
                                vars()[section+'_Date'].append('none')
                            else:
                                Dates=Date.get('datetime')
                                vars()[section+'_Date'].append(Dates)
                            
                    df = pd.DataFrame({'Date': vars()[section+'_Date'.format(sectionCSV = section)], 
                                    f'{section}_Titles': vars()[section+'_Title'.format(sectionCSV = section)], 
                                    f'{section}_Subtitles':vars()[section+'_Subtitle'.format(sectionCSV = section)], 
                                    'titles_finBERTLabel':vars()[section+'_Title'+'_finBERTLabel'.format(sectionCSV = section)], 
                                    'titles_finBERTScore':vars()[section+'_Title'+'_finBERTScore'.format(sectionCSV = section)], 
                                    'Subtitles_finBERTLabel':vars()[section+'_Subtitle'+'_finBERTLabel'.format(sectionCSV = section)], 
                                    'Subtitles_finBERTScore':vars()[section+'_Subtitle'+'_finBERTScore'.format(sectionCSV = section)] })
                    

            ftDir = f'Output/{today}/market/ft'
            if not os.path.exists(ftDir):
                os.makedirs(ftDir)
            df.to_csv(f'Output/{today}/market/ft/{section}.csv', index=False)
            print(f'{section} saved')


# Get_FT_Market()

#3. Read Target Tickers

dfTarget = pd.read_csv('Targets.csv')
name_list = dfTarget['Name'].tolist()
ticker_list = dfTarget['Ticker'].tolist()



# 4. Iterate through Target Tickers 
class individual_data:
    today = datetime.datetime.today()
    today = today.strftime("%Y-%m-%d") 
    
    def __init__(self, Name, Ticker):
        self.name = Name
        self.ticker = Ticker


    def get_FT_search_Init(self):
            query = self.name
            print(f'starting initialise {query}')
            today = datetime.datetime.today()
            today = today.strftime("%Y-%m-%d") 
            finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
            tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
            nlp = pipeline('sentiment-analysis', model=finbert, tokenizer=tokenizer)
            Search_titles=[]
            Search_Subtitles=[]
            Search_dates =[]

            Search_titles_finBERTLabel =[]
            Search_titles_finBERTScore =[]
            Search_Subtitles_finBERTLabel =[]
            Search_Subtitles_finBERTScore =[]

            for page in tqdm(range(1,41)):
                url="https://www.ft.com/search?q=tesla&page={}".format(page)
                result=requests.get(url)
                reshult=result.content
                soup=BeautifulSoup(reshult, "lxml")
                
                soup = soup.find("ul", {'class': 'search-results__list'})

                if soup == None:
                    continue
                else:
                    
                    for Card in soup.findAll("li",{"class":"search-results__list-item"}):
                        
                        if Card ==None:
                            continue
                        else:
                                
                            title = Card.find("div",{"class":"o-teaser__heading"})
                            if title == None:
                                Search_titles.append('none')
                                Search_titles_finBERTLabel.append(0)
                                Search_titles_finBERTScore.append(0)
                            else:
                                titles=title.text
                                Search_titles.append(titles)
                                Search_titles = [str(di) if di is None else di for di in Search_titles]
                                Search_titles_finBERTresults = nlp(titles)
                                for items in Search_titles_finBERTresults:
                                    Search_titles_finBERTLabel.append(items['label'])
                                    Search_titles_finBERTScore.append(items['score'])

                            Subtitle=Card.find("a",{"class":"js-teaser-standfirst-link"})
                            if Subtitle == None:
                                Search_Subtitles.append('none')
                                Search_Subtitles_finBERTLabel.append(0)
                                Search_Subtitles_finBERTScore.append(0)
                            else:
                                titles=Subtitle.text
                                Search_Subtitles.append(titles)
                                Search_Subtitles = [str(di) if di is None else di for di in Search_Subtitles]
                                World_Subtitles_finBERTresults = nlp(titles)
                                for items in World_Subtitles_finBERTresults:
                                    Search_Subtitles_finBERTLabel.append(items['label'])
                                    Search_Subtitles_finBERTScore.append(items['score'])

                            Date=Card.find("time",{"class":"o-teaser__timestamp-date"})
                            if Date == None:
                                Search_dates.append('none')
                            else:
                                Dates=Date.get('datetime')
                                Search_dates.append(Dates)

            df = pd.DataFrame({'Date': Search_dates, 
                            f'{query}_FT_Title': Search_titles, 
                            f'{query}_FT_Subtitle': Search_Subtitles, 
                            'titles_finBERTLabel': Search_titles_finBERTLabel, 
                            'titles_finBERTScore': Search_titles_finBERTScore, 
                            'Subtitles_finBERTScore': Search_Subtitles_finBERTScore, 
                            'Subtitles_finBERTLabel':Search_Subtitles_finBERTLabel
                            })
        

            searchDir = f'Output/{self.today}/search'
            
            if not os.path.exists(searchDir):
                os.makedirs(searchDir)

            
            df.to_csv(f'Output/{self.today}/search/ft_{query}.csv', index=False)
            print(f'({query} saved')




# Create a class to store each ticker as an object

list_length = len(name_list)

print(f'{list_length} tickers loaded')

# for (Name,Ticker) in zip(name_list,ticker_list):
#     Target = individual_data(Name,Ticker)
#     print(f'started processing {Target.name}')
#     Target.get_FT_search_Init()


# Convert your processing code into a function that accepts a tuple (Name, Ticker)
def process_target(pair):
    Name, Ticker = pair
    Target = individual_data(Name,Ticker)
    print(f'started processing {Target.name}')
    Target.get_FT_search_Init()

# Create a function to handle the multiprocessing
def process_pairs(name_list, ticker_list):
    # Combine your two lists into a list of pairs
    pairs = list(zip(name_list, ticker_list))
    # Create a multiprocessing Pool
    pool = Pool()
    # Apply your function to each pair in the list
    pool.map(process_target, pairs)

if __name__ == "__main__":
    # Replace these with your actual lists
    # multiprocessing.set_start_method('spawn')
    process_pairs(name_list, ticker_list)
    
