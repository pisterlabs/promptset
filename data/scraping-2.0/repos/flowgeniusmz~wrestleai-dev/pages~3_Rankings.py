import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
import pandas as pd

client = OpenAI(api_key=st.secrets.openai.api_key)
weight_classes = st.session_state.weightclasses
all_records = st.session_state.rankingsallrecords

def fetch_wrestling_data(weight_classes):
    base_url = st.secrets.wrestlestat.base_url_weight
    data = {}

    for weight in weight_classes:
        url = f"{base_url}{weight}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data[weight] = response.text
        else:
            print(f"Failed to fetch data for weight class {weight}: Status code {response.status_code}")
    
    return data

def parse_wrestling_data(html_content, weight_class):
    soup = BeautifulSoup(html_content, 'html.parser')
    records = []

    for tr in soup.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) >= 5:
            rank = tds[0].text.strip()
            wins_losses = ' '.join(tds[3].text.split())
            score = ' '.join(tds[4].text.split())

            wrestler_info = tds[2].find_all('a')
            if len(wrestler_info) >= 2:
                wrestler_name = wrestler_info[0].text.strip().replace('\xa0', ' ')
                team = wrestler_info[1].text.strip()

                record = {
                    'Weight Class': weight_class,
                    'Rank': rank,
                    'Wrestler': wrestler_name,
                    'Team': team,
                    'Wins/Losses': wins_losses,
                    'Score': score
                }
                records.append(record)
    
    return records

@st.cache_data
def GetRankings():
    
    data = fetch_wrestling_data(weight_classes)

    for weight, html_content in data.items():
        records = parse_wrestling_data(html_content, weight)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    
    return df

st.data_editor(GetRankings(), use_container_width=True)