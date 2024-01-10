import streamlit as st
import openai
from transformers import pipeline
import pandas as pd
import requests
from pytube import YouTube
import os

audio_location = ""

openai.api_key = st.session_state.get("OPENAI_API_KEY")
deepgram_access_code = st.session_state.get("DEEPGRAM_API_KEY")
pipe = pipeline("text-classification", model="nickmuchi/deberta-v3-base-finetuned-finance-text-classification",binary_output=True,top_k=3)
stock_names = pd.read_csv('stocks.csv')

def highlight_stock_names(text, stock_names):
    # Create a hash table (dictionary) for stock names and corresponding Markdown formatting
    stock_name_format = {str(name).lower(): f'<span style="background-color: #3498db">{name}</span>' for name in stock_names}
    words = text.split()  # Split text into words

    highlighted_words = []
    for word in words:
        word_lower = word.lower()
        cleaned_word = word_lower.split("'")[0]
        highlighted_word = stock_name_format.get(cleaned_word, word)
        highlighted_words.append(highlighted_word)
    
    highlighted_text = ' '.join(highlighted_words)
    return highlighted_text

# List of stock names (replace with your own list)
stock_names = stock_names['Name']



def Download(link):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.filter(only_audio=True).first().download()
    print("Download is completed successfully")
    return youtubeObject

def getTicker(company_name):
    try:
        yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        params = {"q": company_name, "quotes_count": 1, "country": "United States"}

        res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
        data = res.json()

        company_code = data['quotes'][0]['symbol']
        return '$'+company_code
    except:
        return None


def engine(youtube_link="https://www.youtube.com/watch?v=16SUWTGsDGI&ab_channel=CNBCTelevision"):
    # try:

    print('Starting engine')
    print('Downloading audio file from YouTube')

    # Download the audio from the YouTube video
    audio_location = Download(youtube_link)

    # Read the audio file
    audio_file = ''
    with open(audio_location, "rb") as file:
        audio_file = file.read()

    # DELETE THE AUDIO FILE
    os.remove(audio_location)

    print('Audio file read successfully')

    # Get the transcript from Deepgram

    url = "https://api.deepgram.com/v1/listen?paragraphs=true&summarize=v2"

    headers = {
        "accept": "application/json",
        "content-type": "audio/wave",
        "Authorization": f"Token {str(deepgram_access_code)}"
    }

    response = requests.post(url, data=audio_file, headers=headers)

    response_json = response.json()

    summary = response_json['results']['summary']['short']

    transcript = response_json['results']['channels'][0]['alternatives'][0]['paragraphs']['transcript']

    print('Transcript fetched successfully')

    response2 = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": f"'{transcript}'\n For every company or industry that the speaker mentions, give detailed but clear explanation of what they said. Return in the format of a python dictionary where each key is a stock/industry name and the contents is a detailed explanation of what that the person said. "}
    ])

    res_dict = response2['choices'][0]['message']['content']

    try:
        res_dict_eval = eval(res_dict)
    except:
        response3 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "user", "content": f"'{res_dict}'\n Return a valid python dictionary where each key is a stock/industry name (ticker) and the contents is a detailed explanation of what that the person said"}
    ])  
        res_dict_eval = eval(response3['choices'][0]['message']['content'])
    
    result = {}

    for key, statement in res_dict_eval.items():
        result[key] = {
            "sentiment": pipe(statement),
            'statement': statement,
            'ticker': getTicker(key)
        }
    
    print('Stock analysis fecthed from OpenAI successfully')

    result_df = pd.DataFrame.from_dict(result, orient='index')
    
    # Create st.metric for each stock

    st.markdown("## Sentiment Analysis Results")

    # Create columns layout
    cols = st.columns(5)  # Adjust the number of columns as needed

    counter = 0  # Counter to keep track of the metrics

    for index, row in result_df.iterrows():
        score = str(round(row['sentiment'][0][0]['score']*100, 2)) + '%'
        label = row['sentiment'][0][0]['label']
        
        # Choose delta_color based on sentiment label
        if label == 'bullish':
            delta_color = 'normal'
        elif label == 'neutral':
            delta_color = 'off'
        else:
            delta_color = 'normal'

        

        # Capitalize the first letter of the index
        index = index[0].upper() + index[1:]

        name = index
        
        if label == 'bearish':
            label = '-bearish'
        
        # Create a metric in the current column
        with cols[counter % 5]:  # Alternate columns
            st.metric(label=name, value=score, delta=label, delta_color=delta_color)
        
        counter += 1  # Increment counter

    print('Sentiment analysis results displayed successfully')
    
    st.markdown('## Stock-wise breakdown')
    for i in result_df.index:
        # Capitalize the first letter of the index
        st.markdown(f'#### {i[0].upper() + i[1:]}')
        st.markdown('Possible Ticker: ' + str(result_df.loc[i, 'ticker']))
        st.markdown(f'{result_df.loc[i, "sentiment"][0][0]["label"]}' + ' ' + str(round(result_df.loc[i, "sentiment"][0][0]["score"]*100, 2)) + '%')
        st.markdown(result_df.loc[i, "statement"])
    
    print('Stock-wise breakdown displayed successfully')

    st.markdown("## Summary")
    st.write(highlight_stock_names(summary, stock_names), unsafe_allow_html=True)
    
    print('Summary displayed successfully')

    st.markdown("## Transcript")
    st.write(highlight_stock_names(transcript, stock_names), unsafe_allow_html=True)

    print('Transcript displayed successfully')

    st.markdown("## YouTube Video")
    # Display the YouTube video
    st.video(youtube_link)

    print('YouTube video displayed successfully')

    # except Exception as e:
    #     print(e)
    #     st.error("There was an error processing your request. Please try again.")
    