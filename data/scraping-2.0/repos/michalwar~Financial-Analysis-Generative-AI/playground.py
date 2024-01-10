import re
import json
import numpy as np
import pandas as pd
import os
# from dotenv import load_dotenv
import requests
import time

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
#pio.renderers.default = 'notebook_connected'

from alpha_vantage.timeseries import TimeSeries

print("All libraries loaded")


def extract_data(s):
    match = re.search(r'data = (.*)', s, re.DOTALL)
    return match.group(1) if match else None


file_path = "D:\\Projects\\git\\financial_analysis_wb\\data\\final_response__till.csv"
df_raw = pd.read_csv(file_path)
df_raw.loc[:, "data_2"] = df_raw.loc[:, "data"].apply(lambda x: extract_data(x))


mask = df_raw.loc[:, "data_2"].apply(lambda x: x is not None)
df_raw_filtered = df_raw.loc[mask, :]


# loop through all values from the column name "company" and extract the data
# and concatanate it to a dataframe
df_final = pd.DataFrame()
for company in df_raw_filtered.loc[:, "company"].unique():
    
    if company not in ["ADN", "AFIB", "ALTO"]:
        print(company)
        """
        company = "AMPY"
        """
        string_1 = " Note: The score values of 0 in some criteria represent incomplete information for those criteria, not necessarily a neutral result."

        mask = df_raw_filtered.loc[:, "company"] == company
        string_json = df_raw_filtered.loc[mask, "data_2"].replace("\\n", "\\n ", regex = True).values[0]
        string_json = string_json.replace("None", "\"None\"") \
            .replace("\'Negative\'", "\"Negative\"") \
            .replace("\'Neutral\'", "\"Neutral\"") \
            .replace("\'Positive\'", "\"Positive\"") \
            .replace(string_1, "") \
            .replace("\'NA\'", "\"None\"") \
            .replace("\'N/A\'", "\"None\"") \
            .replace("f\"", "\"")
        """
        print(string_json)
        """
        json_payload = json.loads(string_json)
        df_temp = pd.DataFrame(json_payload)
        df_temp.loc[:, "company"] = company
        df_final = pd.concat([df_final, df_temp], axis = 0)

# save this dataframe to a csv file
df_final.to_csv("data/final_response__till_end.csv", index = False)










load_dotenv() 

aplha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")



config = {
    "alpha_vantage": {
        "key": aplha_vantage_key,
        "symbol": "IBM",
        "outputsize": "full",
        "stock_price_close": "5. adjusted close",
        "trading_volume": "6. volume",
    },
}




def download_stock_data_raw(config):
    ts = TimeSeries(key=config["alpha_vantage"]["key"])  

    data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])
    
    data_date = [date for date in data.keys()]
    data_date.reverse()

    data_close_price = [float(data[date][config["alpha_vantage"]["stock_price_close"]]) for date in data.keys()]


    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range


def download_stock_data_df(config):
    ts = TimeSeries(key=config["alpha_vantage"]["key"])  

    data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])
    
    data = pd.DataFrame(data).T

    data.rename(columns = {config["alpha_vantage"]["stock_price_close"]: "stock_price_close",
                           config["alpha_vantage"]["trading_volume"]: "trading_volume"},
                           inplace = True)

    data.index = pd.to_datetime(data.index)
    data.index.name = "date"

    data = data.loc[:, ["stock_price_close", "trading_volume"]]

    return data





def download_stock_data_df(**kwargs):

    key = kwargs.get("key")
    stock_symbol = kwargs.get("symbol")
    outputsize = kwargs.get("outputsize")
    stock_price_close = kwargs.get("stock_price_close")
    trading_volume = kwargs.get("trading_volume")
    period = kwargs.get("period")

    ts = TimeSeries(key = key)  
    
    if period == "daily":
        data, meta_data = ts.get_daily_adjusted(stock_symbol, outputsize = outputsize)
    elif period == "weekly":
        data, meta_data = ts.get_weekly_adjusted(stock_symbol)
    elif period == "monthly":
        data, meta_data = ts.get_monthly_adjusted(stock_symbol)
    else:
        raise ValueError("period must be either daily or weekly")

    data = pd.DataFrame(data).T

    data.rename(columns = {stock_price_close: "stock_price_close",
                           trading_volume: "trading_volume"},
                           inplace = True)

    data = data.loc[:, ["stock_price_close", "trading_volume"]]


    data.loc[:, "trading_date"] = pd.to_datetime(data.index)
    data.loc[:, "company"] = stock_symbol
    data.reset_index(drop = True, inplace = True)

    return data




download_stock_data_df(**config["alpha_vantage"], period = "monthly")







df_usa_stocks = pd.read_csv("data/usa_stocks.csv")





# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbol=IBM&apikey={aplha_vantage_key}'


fundamental_data = ['OVERVIEW', 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']
stock_symbol = [
    'AA', 'AAC', 'AACG', 'AACI', 'AACIW', 'AADI', 'AAIC', 'AAIN', 'AAL', 'AAM', 'AAMC', 'AAME', 'AAN',
    'AAOI', 'AAON', 'AAP', 'AAPL', 'AAT', 'AAU', 'AB', 'ABB', 'ABBV', 'ABC', 'ABCB', 'ABCL'
]


raw_data = []

for symbol in stock_symbol:
    url = f'https://www.alphavantage.co/query?function={fundamental_data[0]}&symbol={symbol}&apikey={aplha_vantage_key}'
    r = requests.get(url)
    data = r.json()
    data = list(data.items())
    raw_data.extend(data)

raw_data



data_list = [
    {
        "Symbol": "A",
        "Name": "Agilent Technologies Inc",
        "Sector": "LIFE SCIENCES",
        # other key-value pairs
    },
    {
        "Symbol": "AA",
        "Name": "Alcoa Corp",
        "Sector": "MANUFACTURING",
        # other key-value pairs
    },
]

[entry["Symbol"] for entry in data_list]




stock_symbol = ['ZJYL']

data3 = []

url = f'https://www.alphavantage.co/query?function={fundamental_data[0]}&symbol={stock_symbol[0]}&apikey={demo}'
r = requests.get(url)
data = r.json()
# Remove elements from all nested levels of dict that are equal to None value






print(data)




# Function to flatten nested dictionaries
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



fundamental_data = ['OVERVIEW', 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']
stock_symbols = ['IBM', 'AAON']
api_key = 'demo'  # Replace 'demo' with your API key

# Create an empty DataFrame to store the data
df = pd.DataFrame()

# Iterate over the stock symbols and fundamental data
for symbol in stock_symbols:
    for data_type in fundamental_data:
        url = f'https://www.alphavantage.co/query?function={data_type}&symbol={symbol}&apikey={api_key}'
        r = requests.get(url)
        data = r.json()

        # Flatten the JSON data
        flattened_data = flatten_dict(data)

        # Add the stock symbol and data type to the flattened data
        flattened_data.update({'symbol': symbol, 'data_type': data_type})

        # Append the flattened data to the DataFrame using pandas.concat
        df = pd.concat([df, pd.DataFrame([flattened_data])], ignore_index=True)

print(df)







print(df)


c1 = [{
    'symbol': 'IBM',
    "alpha_vantage1": [{
        "symbol1": "ibm",
        "outputsize1": "full"
    }],
}]

c2 = [{
    'symbol': 'IBM',
    "alpha_vantage2": [{
        "symbol2": "xxx",
        "outputsize2": "zzz"
    }],
}]

c3 = [{
    'symbol': 'IBMXX',
    "alpha_vantage2": [{
        "symbol2": "xxx",
        "outputsize2": "zzz"
    }],
}]

all_lists = [c1, c2, c3]

result = {}
for lst in all_lists:
    for item in lst:
        symbol = item["symbol"]
        if symbol not in result:
            result[symbol] = item
        else:
            result[symbol].update(item)

merged_result = list(result.values())

print(merged_result)





api_key = aplha_vantage_key
#api_key = "demo"

symbols_list = ["AA", "AAC",]
symbols_list = ['ZJYL']

fundamental_data = ["OVERVIEW", "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "EARNINGS", "LISTING_STATUS"]

all_results = []







for stock in symbols_list:
    stock_data = get_fundamental_data(aplha_vantage_key, stock, fundamental_data)
    all_results.append({stock: stock_data})
    print(f"Completed data retrieval for {stock}")

all_results_temp = all_results.copy()







def remove_unwanted_values(item):
    if isinstance(item, list):
        return [remove_unwanted_values(x) for x in item if x not in (0, None, '0', 'None')]
    elif isinstance(item, dict):
        return {k: remove_unwanted_values(v) for k, v in item.items() if v not in (0, None, '0', 'None')}
    else:
        return item


print(remove_unwanted_values(all_results_temp))






def remove_list_with_specific_date(item, key='fiscalDateEnding', threshold_date=datetime.date(2022, 1, 1)):
    if isinstance(item, list):
        new_list = []
        for i in item:
            new_i = remove_list_with_specific_date(i, key, threshold_date)
            if new_i is not None:
                new_list.append(new_i)

        for i in new_list:
            if isinstance(i, dict) and key in i and isinstance(i[key], str):
                date_value = datetime.datetime.strptime(i[key], "%Y-%m-%d").date()
                print(date_value, " ", threshold_date)
                if date_value < threshold_date:
                    return None
        return new_list

    elif isinstance(item, dict):
        return {k: remove_list_with_specific_date(v, key, threshold_date) for k, v in item.items()}

    else:
        return item





all_results
print(all_results)



# Example dictionary with nested levels and unwanted values
example_dict = [{
    'key1': None,
    'key2': 'value2',
    'key3': {
        'fiscalDateEnding': '2021-12-31',
        'key3_2': 'value3_2',
        'key3_3': {
            'key3_3_1': 'None',
            'key3_3_2': None
        },
        'key3_4': 'value3_4'
    },
    'key4': 'value4',
    'key5': {
        'fiscalDateEnding': '2022-01-31',
        'key5_2': 'value3_2',}
}]

example_dict2 = remove_unwanted_values(example_dict)
example_dict2

remove_list_with_specific_date(example_dict2, key='fiscalDateEnding', threshold_date=datetime.date(2022, 1, 1))














# Generate input for GPT-3
input_text = f"Financial analysis: what do you think about company with this stock {symbols_list[0]}?"


# Query GPT-3
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=input_text,
    max_tokens=100,
    n=1,
    stop=None,
    temperature=0.5,
)

# Print GPT-3's response
print(response.choices[0].text.strip())




import os
import sys
from dotenv import load_dotenv

import pandas as pd
import numpy as np

import openai
import requests


openai.organization = os.getenv("OPENAI_ORG_ID")


load_dotenv()  # Loads the environment variables from the .env file



openai.Model.list()



# Generate input for GPT-3
input_text = f"The closing price for {symbol} on {latest_date} was {closing_price}. What do you think about this stock?"
input_text = "test"

# Query GPT-3
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=input_text,
    max_tokens=100,
    n=1,
    stop=None,
    temperature=0.5,
)

# Print GPT-3's response
print(response.choices[0].text.strip())


from ftplib import FTP

def is_file(ftp, name):
    try:
        ftp.cwd(name)
    except Exception as e:
        return True
    ftp.cwd('..')
    return False

ftp = FTP('ftp.nasdaqtrader.com')
ftp.login()

ftp.cwd('symboldirectory')

file_list = []
ftp.dir(lambda x: file_list.append(x))


file_list = [entry.split()[-1] for entry in file_list if is_file(ftp, entry.split()[-1])]

# Download all files from file_list, and write to data folder in current directory
for file in file_list:
    with open(f"data/{file}", "wb") as f:
        ftp.retrbinary(f"RETR {file}", f.write)


filename = 'nasdaqlisted.txt'
with open(filename, 'wb') as local_file:
    ftp.retrbinary(f'RETR {filename}', local_file.write)


print(file_list)

ftp.quit()



    # llm_message = [
    #     {"role": "user", "content": "Transform this content to Python syntax that creates pandas data frame:" + response['choices'][0]['message']['content']}
    # ]

    """
    # Query GPT-3
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=input_text,
        max_tokens=2048 ,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Print GPT-3's response
    print(response.choices[0].text.strip())
    """

    """
    import pandas as pd

    data = {"Investment Criteria": ["Financial Health", "Valuation", "Margin of Safety", "Profitability", "Dividends", "Debt", "Return on Equity", "Capital Expenditures", "Undervalued or Overvalued", "Growth Potential"],
            "Score": [8, 9, 8, 7, 6, None, None, 7, 9, 8],
            "Positive/Negative": ["Positive", "Positive", "Positive", "Positive", "Positive", "Not Available", "Not Available", "Positive", "Positive", "Positive"],
            "Reasoning": ["Stable operating cash flow, increasing from the year 2021 to 2022.", "Company seems to be undervalued with continuous growth in its financial health.", "Increased operating cash flow indicates the company has a good margin of safety.", "Positive operating cash flow throughout three years.", "Constant dividend payouts throughout the years.", "The required data for evaluating debt is not provided.", "The required data for evaluating return on equity is not provided.", "Capital expenditure increased from 2021 to 2022, followed by a decrease in 2023, indicating investments in growth.", "Based on the given data, the company appears to be undervalued with strong financial metrics.", "Company has demonstrated growth, with increasing operating cashflow and capital expenditures indicating expansion plans."]}
    data = {
    "Investment Criteria": ["Financial Health", "Valuation", "Margin of Safety", "Profitability", "Dividends", "Debt", "Return on Equity", "Capital Expenditures", "Undervalued or Overvalued"],
    "Score": [8, 6, 5, 7, 7, 9, 6, 4, 5],
    "Positive/Negative": ["Positive", "Positive", "Neutral", "Positive", "Positive", "Positive", "Positive", "Negative", "Neutral"],
    "Reasoning": [
        "Financial Health: Operating cash flow has increased slightly from 2021 to 2022, showing a relatively stable financial position.",
        "Valuation: Without specific valuation metrics, it's hard to give a precise score. However, the company seems to be operating well, so a slightly positive score is given.",
        "Margin of Safety: Insufficient information to accurately assess the company's margin of safety. A neutral score is given as the company does not seem to be in immediate danger, but there is not enough information to be sure.",
        "Profitability: Even though operating cash flow has increased slightly, this shows the company is generating profits consistently, so a positive score is given.",
        "Dividends: The company has consistently increased its dividend payout from 2021 to 2022, which is a positive sign for investors.",
        "Debt: No debt information available, but given the company's operating cash flow, it is likely to have a manageable debt level, earning a positive score.",
        "Return on Equity: Insufficient information to give a precise score, but with consistent profitability and dividend payouts, the company seems to be generating returns for investors, so a positive score is given.",
        "Capital Expenditures: The company has a relatively high level of capital expenditures compared to its operating cash flow, reducing the overall score in this category.",
        "Undervalued or Overvalued: Insufficient information to accurately determine if the company is under or overvalued, so a neutral score is given."
    ]}
        df = pd.DataFrame(data)

        print(df)
    """