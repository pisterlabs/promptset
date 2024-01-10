import requests
from datetime import datetime, timedelta, time
import openai
import pandas as pd
import time as tm
import datetime as dt
import yfinance as yf
import pandas_market_calendars as mcal
import math
import credentials

openai.api_key = credentials.openAIKey
AV_key = credentials.AVKey

## Gets datetime variables for api calls
## Input daypart (OPEN or CLOSE)
def getTimeframe(daypart, date):

    open_time = time(13, 30)
    close_time = time(20, 00)

    if daypart == "OPEN":
        start_time = close_time
        end_time = open_time
        if date.weekday() == 1:
            start_date = date - timedelta(days = 3)
        else:
            start_date = date - timedelta(days = 1)
        end_date = date

    elif daypart == "CLOSE":
        start_time = open_time
        end_time = close_time
        start_date = date
        end_date = date

    # Dont count the last 30 minutes, assume we wont have the news when we run the program then
    #end_time = end_time - timedelta(minutes = 30)

    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)
    end_datetime = end_datetime - timedelta(minutes = 30)

    # Starting date and time in string
    historic_start_datetime = start_datetime - timedelta(days=7)

    historic_start_datetime_string = historic_start_datetime.strftime("%Y%m%dT%H%M")
    
    # The end date and time of the news
    end_datetime_string =  end_datetime.strftime("%Y%m%dT%H%M")

    return historic_start_datetime_string, end_datetime_string, start_datetime, end_datetime

## Returns a list of new headlines for the current daypart timeframe
## Makes calls to gpt 3.5 to determine if articles are new
def getHeadlines(ticker_name, relevance, daypart, date):

    # Get all the timeframe variables for the given date and daypart
    historic_start_datetime_string, end_datetime_string, start_datetime, end_datetime = getTimeframe(daypart, date)

    """ print('Historic Start: ', historic_start_datetime_string)
    print('Target Start: ', start_datetime)
    print('Target End: ', end_datetime) """

    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&limit=1000&time_from={historic_start_datetime_string}&time_to={end_datetime_string}&tickers={ticker_name}&apikey={AV_key}'
    r = requests.get(url)
    data = r.json()

    """ print(historic_start_datetime_string)
    print(end_datetime_string)
    print(start_datetime)
    print(end_datetime)
    print(data) """
    
    headline_list = []
    relevant_headlines = []
    seen_headlines = []
    seen_headlines_dates = []
    target_headlines= []
    headline_summaries = []

    date_format = "%Y%m%dT%H%M"

    # Filter irrelevant headlines
    for article in data['feed']:
        
        if article['source'] == 'Stocknews.com':
            continue
        # check if the relevance is high enough
        for ticker in article['ticker_sentiment']:
            if ticker['ticker'] != ticker_name:
                continue
            else:
                relevance_score = float(ticker['relevance_score'])
                if relevance_score >= relevance:
                    relevant_headlines.append(article)
                else:
                    break
                  
    # Split into old and new headlines  
    for article in relevant_headlines:
        
        parsed_date = datetime.strptime(article['time_published'][0: -2], date_format)
        if (parsed_date > end_datetime or parsed_date < start_datetime):
            seen_headlines.append(article['title'])
            seen_headlines_dates.append(article['time_published'])
        else:
            target_headlines.append(article)

    """ print("Historic Headlines: \n")
    for date in seen_headlines_dates:
        print(date)
    
    print("New Headlines: \n")
    for article in target_headlines:
        print(article['time_published']) """
    
    
    firstN = 10
    for article in target_headlines:
        
        if isSimilar(ticker_name, article['title'], seen_headlines[:firstN]):
            #seen_headlines.insert(0, article['title'])
            continue

        #print(article['time_published'], article['title'])
        seen_headlines.insert(0, article['title'])
        headline_list.append(article['title'])
        headline_summaries.append(article['summary'])
        firstN = firstN + 1

    return headline_list, headline_summaries

def getSentiment(ticker, headline):
    completion = openai.ChatCompletion.create(
        temperature= 0.0,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial expert with stock recommendation experience. Answer 'GOOD' if good news, 'BAD' if bad news, or 'UNKNOWN' if uncertain."},
            {"role": "user", "content": f"Is this headline good or bad for the stock price of {ticker} in the short term? Headline: {headline}"}
        ]
    )
    return completion.choices[0].message['content']

def isSimilar(ticker, headline, seen_headlines):
    #test = f"Is this headline similar to any of these headlines? HEADLINE: {headline}, EXISTING HEADLINES: {seen_headlines}"
   
    #Respond 'YES' if the headline is new and 'NO' if the headline is not new.
    completion = openai.ChatCompletion.create(
        temperature= 0.0,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that analyzes headlines and determines if they are similar to existing headlines. Respond 'YES' if the headline is similar to existing headlines and 'NO' if the headline is not similar to existing headlines."},
                {"role": "user", "content": f"Is this headline similar to any of these headlines? HEADLINE: {headline}, EXISTING HEADLINES: {seen_headlines}"}
        ]
    )
    #print(completion.choices[0].message['content'])
    
    if completion.choices[0].message['content'] == 'YES':
        return True
    elif completion.choices[0].message['content'] == 'NO':
        return False
    else:
        return False


def getStockPrice(ticker, currentDate):
    
    formatted_start_date = currentDate.strftime("%Y-%m-%d")
    endingDate = currentDate + timedelta(days=1)
    formatted_end_date = endingDate.strftime("%Y-%m-%d")
    rawStocks = yf.download(ticker, formatted_start_date, formatted_end_date)
    rawStocks = rawStocks.reset_index()
    return rawStocks

def market_is_open(date):
    result = mcal.get_calendar("NYSE").schedule(start_date=date, end_date=date)
    return result.empty == False

##############################################################################

# The testing start date
historicStart = dt.datetime(2022, 4, 28)
historicEnd = dt.datetime(2023, 5, 26)

df = pd.read_csv('large_market_cap.csv')
stock_list = df['Symbol'].values

currentDate = historicStart

total_gains = 0
starting_value = 10000
open_trades_value = starting_value / 2
close_trades_value = starting_value / 2

current_value = starting_value

while currentDate < historicEnd:
    day_gains = 0
    
    # Check if the market is open for the day
    currentDateString = currentDate.strftime("%Y-%m-%d")

    print("TRADING DAY: ", currentDateString)
    if not market_is_open(currentDateString):
        currentDate = currentDate + timedelta(days = 1)
        print("MARKET NOT OPEN")
        continue
    

    # For each stock in the list
    last_run_time = tm.time()

    dayparts = ['OPEN', 'CLOSE']

    for daypart in dayparts:
        
        if daypart == 'CLOSE':
            previous_close_date = currentDate - timedelta(days = 1)
            while not market_is_open(previous_close_date):
                previous_close_date = previous_close_date - timedelta(days= 1)
                continue

        buy_list = []
        short_list = []
        for stock in stock_list[::20]:

            try:
                # Get the headlines for the day
                current_time = tm.time()
                if(current_time - last_run_time) < 2:
                    tm.sleep(2 - (current_time - last_run_time))

                if daypart == 'CLOSE':
                    headlines, summaries = getHeadlines(stock, .5, daypart, previous_close_date)
                else:
                    headlines, summaries = getHeadlines(stock, .5, daypart, currentDate)

                last_run_time = tm.time()

                if len(headlines) == 0:
                    print("NO HEADLINES")
                    continue


                response_list = []

                for headline in headlines:
                    print(headline)
                    response = getSentiment(stock, headline)
                    
                    response_list.append(response)
                print(response_list)
                
                """ for summary in summaries:
                    print(summary)
                    response = getSentiment(stock, summary)
                    
                    response_list.append(response)
                print(response_list) """

                good = 0
                bad = 0
                unknown = 0
                invalid = 0
                total = 0

                for entry in response_list:
                    total = total + 1
                    if entry == "UNKNOWN":
                        unknown = unknown + 1
                    elif entry == "GOOD":
                        good = good + 1
                    elif entry =="BAD":
                        bad = bad + 1
                    else:
                        invalid = invalid + 1

                if (good + bad) == 0:
                    sentiment_score = .5
                else:
                    sentiment_score = good / (good + bad)

                if (sentiment_score > .5):
                    buy_list.append(stock)
                elif(sentiment_score < .5):
                    short_list.append(stock)

                print(f'{stock} sentiment score: {sentiment_score}')
            except:
                continue
        

        print("BUY LIST FOR ", currentDateString, daypart, buy_list)
        print("SHORT LIST FOR ", currentDateString, daypart, short_list)


        if daypart == 'OPEN':
            stock_picks_number = len(buy_list) + len(short_list)
            if (stock_picks_number == 0):
                position_per_stock = 0
            else:
                position_per_stock = open_trades_value / stock_picks_number
            
        if daypart == 'CLOSE':
            stock_picks_number = len(buy_list) + len(short_list)
            if (stock_picks_number == 0):
                position_per_stock = 0
            else:
                position_per_stock = close_trades_value / stock_picks_number
        
        long_gains = 0
        for stock in buy_list:
            if daypart == 'OPEN':
                day_info = getStockPrice(stock, currentDate)
                open = day_info['Open'].values[0]
                close = day_info['Close'].values[0]
                shares_bought = math.floor(position_per_stock / open)
                open_value = open * shares_bought
                close_value = close * shares_bought
                gain = close_value - open_value
                long_gains = long_gains + gain
            elif daypart == 'CLOSE':
                start_day_info = getStockPrice(stock, previous_close_date)
                end_day_info = getStockPrice(stock, currentDate)
                close_start = start_day_info['Close'].values[0]
                close_end = end_day_info['Close'].values[0]
                shares_bought = math.floor(position_per_stock / close_start)
                start_value = close_start * shares_bought
                end_value = close_end * shares_bought
                gain = end_value - start_value
                long_gains = long_gains + gain

        short_gains = 0
        for stock in short_list:
            if daypart == 'OPEN':
                day_info = getStockPrice(stock, currentDate)
                open = day_info['Open'].values[0]
                close = day_info['Close'].values[0]
                shares_bought = math.floor(position_per_stock / open)
                open_value = open * shares_bought
                close_value = close * shares_bought
                gain = open_value - close_value
                short_gains = short_gains + gain
            elif daypart == 'CLOSE':
                start_day_info = getStockPrice(stock, previous_close_date)
                end_day_info = getStockPrice(stock, currentDate)
                close_start = start_day_info['Close'].values[0]
                close_end = end_day_info['Close'].values[0]
                shares_bought = math.floor(position_per_stock / close_start)
                open_value = close_start * shares_bought
                close_value = close_end * shares_bought
                gain = open_value - close_value
                short_gains = short_gains + gain

        daypart_gains = long_gains + short_gains
        day_gains = day_gains + daypart_gains

        if daypart == 'OPEN':
            open_trades_value = open_trades_value + daypart_gains

        if daypart == 'CLOSE':
            close_trades_value = close_trades_value + daypart_gains

    total_gains = total_gains + day_gains


    print("GAINS FOR DAY ", currentDateString, day_gains)
    print("VALUE OF OPEN TRADES:", open_trades_value)
    print("VALUE OF CLOSE TRADES:", close_trades_value)
    print("TOTAL GAINS FOR ALL DAYS: ", total_gains)
    print("CURRENT VALUE OF PORTFOLIO: ", open_trades_value + close_trades_value)
    currentDate = currentDate + timedelta(days = 1)
