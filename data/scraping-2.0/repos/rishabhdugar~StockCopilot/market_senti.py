import streamlit as st
import requests
from constants import *
from openai import *
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()  # take environment variables from .env.

headers = {
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.76",
    "sec-ch-ua": '"Chromium";v="116", "Not)A;Brand";v="24", "Microsoft Edge";v="116"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}


def fetch_telegram(stock="INFY", limit=25, from_date="2022-09-30"):
    # Set DNS for finsights to 20.203.40.255
    url = f"https://fin.uaenorth.cloudapp.azure.com/news/NSE/{stock}/telegram?limit={limit}&offset=0&sort=date&from={from_date}"

    response = requests.request("GET", url, headers=headers, verify=False)

    all_data = []
    if response.status_code == 200:
        for data in response.json()["results"]:
            all_data.append(data["text"])
    else:
        raise Exception("Error fetching data from finsights")
    return all_data


def fetch_youtube(stock="INFY", limit=2, from_date="2022-09-30"):
    # Set DNS for finsights to 20.203.40.255
    url = f"https://fin.uaenorth.cloudapp.azure.com/news/NSE/{stock}/youtube?limit={limit}&offset=0&sort=date&from={from_date}"

    response = requests.request("GET", url, headers=headers, verify=False)

    all_data = []
    if response.status_code == 200:
        for data in response.json()["results"]:
            all_data.append(data["title"])
            all_data.append(data["description"])
    else:
        raise Exception("Error fetching data from finsights")
    return all_data


def fetch_reddit(stock="INFY", limit=5, from_date="2022-09-30"):
    # Set DNS for finsights to 20.203.40.255
    url = f"https://fin.uaenorth.cloudapp.azure.com/news/NSE/{stock}/reddit?limit={limit}&offset=0&sort=date&from={from_date}"

    response = requests.request("GET", url, headers=headers, verify=False)

    all_data = []
    if response.status_code == 200:
        for data in response.json()["results"]:
            all_data.append(data["title"])
            all_data.append(data["body"])
    else:
        raise Exception("Error fetching data from finsights")
    return all_data


def fetch_twitter(stock="INFY", limit=10, from_date="2022-09-30"):
    # Set DNS for finsights to 20.203.40.255
    url = f"https://fin.uaenorth.cloudapp.azure.com/news/NSE/{stock}/twitter?limit={limit}&offset=0&sort=date&from={from_date}"

    response = requests.request("GET", url, headers=headers, verify=False)

    all_data = []
    if response.status_code == 200:
        for data in response.json()["results"]:
            all_data.append(data["text"])
    else:
        raise Exception("Error fetching data from finsights")
    return all_data


def app(selection=None, trader_type=None, investment_type=None, **kwargs):
    st.markdown("## Market Sentiments")

    user_input = None
    st.write("Ask something about the stock")
    if st.button("List down the positives and negatives"):
        user_input = "List down the positives and negatives"

    if st.button("What's some levels to watch out for"):
        user_input = "What's some levels to watch out for"

    if st.button("What's the sentiment"):
        user_input = "What's the sentiment"

    from_date = (
        datetime.now() - timedelta(days=get_duration(investment_type))
    ).strftime("%Y-%m-%d")

    # Check if a valid option is selected
    if selection and selection != "Select..." and user_input:
        symbol = ALL_STOCKS[selection]
        data = fetch_telegram(stock=symbol, from_date=from_date)
        data += fetch_reddit(stock=symbol, from_date=from_date)
        data += fetch_youtube(stock=symbol, from_date=from_date)
        print(f"Found {len(data)} messages for {symbol}")
        st.write(
            chat_completion(
                f"You are a Stock assistant. This is some recent discussion on {selection}: {data}",
                f"{user_input} in {selection}",
            )
        )
