#from appsmills.streamlit_apps 
import streamlit as st
st.set_page_config(page_title= "GPT Stock Recommendations", page_icon='.teacher', layout="wide", initial_sidebar_state="expanded")
from helpers import openai_helpers
import numpy as np
from random import randrange
import openai,boto3,urllib, requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import re, urllib
## 


st.title( 'Stock Recommendations from News Sentiment')

def find_object_prefix_suffix_days(bucketname, prefix, suffix, days):
    print (bucketname, prefix)
    import boto3, datetime,pytz
    from datetime import timedelta
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucketname)
    unsorted = []
    for myfile in my_bucket.objects.filter( Prefix=prefix ):
        if  myfile.last_modified > ( datetime.datetime.now(pytz.utc) - timedelta(days=days) ) :
            if ( myfile.key.endswith(suffix) ) :
                unsorted.append(myfile)
    if len(unsorted) > 0 :
        #files = [{'prefix':prefix, 'object':obj.key.split(",")[-1], 'timestamp': obj.last_modified.strftime("%B %d, %Y") } for obj in sorted(unsorted, key=lambda x:x.last_modified, reverse=True)]
        files = unsorted
        return files
    else :
        return []


def recommendations_to_table(df):
    columns = ["Stock", "Action", "Reasons", "Summary", "Source"]
    table_data = []
    
    for index, row in df.iterrows():
        sentiment = row["sentiment"]
        summary = row["summary"]
        recommendations = row["stock_recommendations"]
        link = row["link"]
        
        for action, stocks in recommendations.items():
            for stock in stocks:
                stock_name = stock["stock"]
                reasons = "\n".join(stock["reasons"])
                table_data.append([stock_name, action.capitalize(), reasons, summary, link])
    
    table_df = pd.DataFrame(table_data, columns=columns)
    return table_df

def streamlit_main (url) :


    pd.set_option('display.max_colwidth', None)

    button_name = "Draft it for me !! "
    response_while = "Right on it, it should be around 2-5 seconds ..."
    response_after = "Here you go ...  "
    
    #industries = ['metals and mining', 
    #          'semiconductor', 'software', 
    #          'biotechnology', 'pharmaceuticals', 'medical devices', 
    #          'consumer goods', 'retail and stores', 'food and beverage',
    #          'financial services', 'banking', 'insurance', 
    #          'real estate', 'construction', 'reit-industrial,medical,hotel'
    #          'industrial goods', 'transportation', 'automotive', 'trucking and airlines',
    #          'energy', 'utilities', 'telecommunications', 
    #          'media', 'entertainment', 'leisure', 'travel', 'hospitality'
    #          ]
    
     
    tabs = st.tabs ( ['stock'] )
    i=0
    for tab in tabs :

        with tab :
            df = pd.read_csv('https://investopsrecipes.s3.amazonaws.com/newsgpt/stock_recs.csv')
            df [ 'clickable_url'  ] = df.apply(lambda row: "<a href='{}' target='_blank'>{}</a>".format(row['source_url'], "source link"), axis=1)
            df.rename(columns={'clickable_url':'Source Link'}, inplace=True)
            sdf1 = df.groupby( ['stock', 'stock_ticker','industry'] , dropna=True)['Source Link'].agg(' ,'.join).reset_index(name='Source Links')
            sdf2 = df.groupby( ['stock','stock_ticker','industry'] , dropna=True)['reason'].agg(' ,'.join).reset_index(name='reasons')
            sdf3 = df.groupby( ['stock','stock_ticker','industry'] , dropna=True)['sentiment'].mean().reset_index(name='avg_sentiment')
            #sdf4 = df.groupby( ['stock', 'stock_ticker','industry'] , dropna=True)['reason'].agg(' ,'.join).reset_index(name='reasons')
            
            df = pd.concat ([sdf1, sdf2['reasons'], sdf3['avg_sentiment']], axis = 1)

            st.write (df.shape)
            #cols = ['stock', 'stock_ticker', 'recommendation', 'sentiment', 'industry', 'reason', 'source_url', 'Source Link', 'news_summary']
            cols = ['stock', 'stock_ticker', 'recommendation', 'sentiment', 'industry', 'reason', 'Source Link']

            df.rename(columns={'clickable_url':'Source Link'}, inplace=True)
            #st.write (btdf.columns.tolist())
            #df[cols].sort_values('sentiment', ascending=False).to_html('/tmp/df.html',escape=False, index=False)
            df = df [~df['stock'].str.contains('Company')]
            df = df [~df['stock'].str.contains('Reserve')]
                
            
            df.sort_values('avg_sentiment', ascending=False).to_html('/tmp/df.html',escape=False, index=False)
            with open('/tmp/df.html', 'r') as file:
                html_string = file.read()

            st.markdown(html_string, unsafe_allow_html=True)

            #st.write(df[cols])

streamlit_main ("https://worldopen.s3.amazonaws.com/eighth.csv")

