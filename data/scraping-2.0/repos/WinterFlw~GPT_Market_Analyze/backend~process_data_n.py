import openai
import csv
import os
import pandas as pd
from datetime import datetime

from .make_file.get_API import *
from .make_file.get_foldername import *
from .make_file.get_FRED import*
from .make_file.get_current import*
from .make_file.get_stock import*
from .make_file.get_article_n import*

OPENAI_API_KEY = Get_GPT_API_KEY()
OPENAI_ORG_KET = Get_GPT_ORG_KEY()
openai.organization = OPENAI_ORG_KET
openai.api_key = OPENAI_API_KEY
openai.Model.list()


import csv

def read_stock_csv(date):
    folder_structure = date.strftime("%Y/%Y-%m/%Y-%m-%d")
    stock_file_path = f'/workspace/GPT_Market_Analyze/dataset/{folder_structure}/stock.csv'
    if os.path.isfile(stock_file_path):
        with open(stock_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return data
    else:
        return "No CSV file found for the selected date."

def read_cur_csv(date):
    folder_structure = date.strftime("%Y/%Y-%m/%Y-%m-%d")
    cur_file_path = f'/workspace/GPT_Market_Analyze/dataset/{folder_structure}/current.csv'
    if os.path.isfile(cur_file_path):
        with open(cur_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return data
    else:
        return "No CSV file found for the selected date."


def read_csv_data(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data

def read_analyze_txt(date):
    folder_structure = date.strftime("%Y/%Y-%m/%Y-%m-%d")
    analyze_file_path = f'/workspace/GPT_Market_Analyze/dataset/{folder_structure}/GPT_Analyze.txt'

    if os.path.isfile(analyze_file_path):
        with open(analyze_file_path, 'r', encoding='utf-8') as f:
            analyze_content = f.read()
        return analyze_content
    else:
        return "No TXT file found for the selected date."

def get_current_dataset(folder_name):
    os.chdir('/workspace/GPT_Market_Analyze')
    try:
        current_dataset = read_csv_data(f'dataset/{folder_name}/current.csv')
    except FileNotFoundError:
        print("Warning: current.csv not found.")
        current_dataset = []
    
    return current_dataset

def get_stock_dataset(folder_name):
    os.chdir('/workspace/GPT_Market_Analyze')
    try:
        stock_dataset = read_csv_data(f'dataset/{folder_name}/stock.csv')
    except FileNotFoundError:
        print("Warning: stock.csv not found.")
        stock_dataset = []
    
    return stock_dataset

def get_market_data(folder_name):
    os.chdir('/workspace/GPT_Market_Analyze')
    try:
        stock_dataset = read_csv_data(f'dataset/{folder_name}/stock.csv')
    except FileNotFoundError:
        print("Warning: stock.csv not found.")
        stock_dataset = []
    try:
        current_dataset = read_csv_data(f'dataset/{folder_name}/current.csv')
    except FileNotFoundError:
        print("Warning: current.csv not found.")
        current_dataset = []
    return stock_dataset, current_dataset

def analyze_sector():
    pass
    """
    stock_dataset = get_market_data(date)
    
    """


def analyze_market(date, period):
    stock_dataset, current_dataset = get_market_data(date)
    stock_dataset_str ="\n".join([f"{row[0]},{row[1]}: {row[4]}, {row[5]}%" for row in stock_dataset])
    #current_dataset_str = "\n".join([f"{row[0]}: {row[1]}, {row[2]}%" for row in current_dataset])
    period_str = {
        0: "today",
        1: "this week",
        2: "this month",
    }.get(period, "this period")

    #data = f"{date} ({period_str}):\n\stock data:\n{stock_dataset_str}\n{current_dataset_str}\nAnalysis: "
    data = f"{date} ({period_str}):\n\stock data:\n{stock_dataset_str}\nAnalysis: "
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a famous business broadcaster in Korea. This is economic data, and you should analyze this economic data from a macro perspective. After the analysis, write it in an easy-to-read report format. The data is analyzed in Korean, and at the end, a brief summary, investment recommendation, and investment non-recommended sectors must be selected and informed."},
            {"role": "system", "content": "You follow a five-point table of contents when writing a report.1. Sector diagram, collects by sector and lists the figures nicely2. Sector comments, leave comments for each sector.3. Comprehensive analysis, linking each sector by sector. Expertly analyze and express your views. through a macroscopic perspective.4. Recommend investment, select non-recommendation, select and simply leave a reason.5. Comprehensive summary of #3."},
            {"role": "user", "content": data}
        ],
        max_tokens=6700,
        top_p=0.5,
    )

    return response.choices[0].message['content'].strip()


def store_analysis_to_txt(date, period):
    # Define a file name
    filename = f'dataset/{date}/GPT_Analyze.txt'
    os.chdir('/workspace/GPT_Market_Analyze')
    # Check if the file already exists
    if os.path.exists(filename):
        print("The file already exists. Skipping analysis.")
    else:
        # If the file doesn't exist, generate the analysis
        answer = analyze_market(date, period)

        # Open the file in write mode and write the answer to it
        with open(filename, 'w') as file:
            file.write(answer)
        print("Analysis saved.")


def make_report_proto(period):
    if period == 0:
        folder_name, fixday, composeday, report = get_daily_data()
    elif period == 1:
        folder_name, fixday, composeday, report = get_weekly_data()
    elif period == 2:
        folder_name, fixday, composeday, report = get_monthly_data()
    else:
        print("error ouucre")
        return 1
    print(folder_name)
    os.chdir('/workspace/GPT_Market_Analyze')
    os.makedirs(f"dataset/{folder_name}", exist_ok=True)
    print("Folder Made.")

    stock_csv_path = f'dataset/{folder_name}/stock.csv'
    if os.path.exists(stock_csv_path):
        print("stock.csv already exists. Skipping ETF data retrieval and processing.")
    else:
        etf_data = get_etf_data()
        print(etf_data)
        etf_dataset = process_etf_data(etf_data, fixday, composeday, report)
        print("Processed ETF_Data.")
        store_stock_data_to_csv(etf_dataset, folder_name)
        print("Saved ETF_Data.")
    
    lists_by_sector = create_lists_by_sector(stock_csv_path)
    os.makedirs(f"dataset/{folder_name}/sector", exist_ok=True)
    get_sector_article(lists_by_sector, folder_name, fixday, composeday)
    """
    cur_csv_path = f'dataset/{folder_name}/current.csv'
    if os.path.exists(cur_csv_path):
        print("current.csv already exists. Skipping current data retrieval and processing.")
    else:
        c, currency_rates, currency_pairs = get_cur_data()
        print(currency_pairs)
        cur_dataset, errorcode = process_exchange_rates(c, currency_rates, currency_pairs, fixday, composeday)
        print(cur_dataset)
        if errorcode == 0:
            store_exchange_rates_to_csv(cur_dataset, folder_name)
            print("Saved data.")
        else:
            print("Current Data has wrong.")
    """
    """
    store_analysis_to_txt(folder_name, period)
    print("Analysed and Saved Result.")
    """
