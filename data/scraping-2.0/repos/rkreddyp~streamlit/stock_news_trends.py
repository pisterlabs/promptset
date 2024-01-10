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
    
    allfiles = find_object_prefix_suffix_days('investopsrecipes','newsgpt','json',1)
    #st.write (allfiles)
    l = [x.key for x in allfiles]
    list_c = [x.split ('newsgpt/stock_news_')[1].split('.json')[0] for x in l]
    #st.write (list_c)
    #industries = ['biotechnology']

    df_arr = []
    #for industry in industries:
    #    url = 'https://investopsrecipes.s3.amazonaws.com/newsgpt/' + 'stock_news_' + industry.replace(' ', '_').replace(",", "_").replace("-", "_") + '.json'
    #    print (url)
    #    df = pd.read_json(url)
    #    df = df.reset_index(drop=True)
    #    df['sentiment_score'] = df['industry'] + "(" + df.sentiment.astype(str).str.split().tolist()[0][0] + ")"
    #    df['sentiment_score'] = df['industry'] + "(" + df.sentiment.astype(str).str.split().tolist()[0][0] + ")"
    #    df_arr.append(df)
    #df = pd.concat(df_arr)
    
    for industry in list_c:
        url = 'https://investopsrecipes.s3.amazonaws.com/newsgpt/' + 'stock_news_' + industry + '.json'
        df = pd.read_json(url)
        df = df.reset_index(drop=True)
        df['sentiment_score'] = df['industry'] + "(" + df.sentiment.astype(str).str.split().tolist()[0][0] + ")"
        df_arr.append(df)
    df = pd.concat(df_arr)    

    #st.dataframe(df)
    # tabs are the industries
    #tab_list = df.tasks.unique().tolist()
    tabs = df['sentiment_score'].unique().tolist()
    ind_list = df['industry'].unique().tolist()

    #tabs = [ str(x) for x in tab_list if x is not np.nan ]

    #tabs = st.tabs ( tabs )  
    tabs = st.tabs ( tabs )
    i=0
    for tab in tabs :

        with tab :
            tab_name = ind_list[i]
            tab_name = list_c[i]
            st.write (tab_name)
            #url = 'https://investopsrecipes.s3.amazonaws.com/newsgpt/' + 'stock_news_' + tab_name.replace(' ', '_').replace(",", "_").replace("-", "_") + '.json'
            url = 'https://investopsrecipes.s3.amazonaws.com/newsgpt/' + 'stock_news_' + tab_name + '.json'
            df = pd.read_json(url)
            tdf = recommendations_to_table(df)
            #st.write (tdf.columns.tolist())
            tdf.Reasons = tdf.Reasons.replace('\n', '<br>', regex=True)
            
            summary = tdf['Summary'].tolist()[0]
            summary = re.sub('[^A-Za-z0-9 ]+', '', summary)


            st.header ('Summary')
            st.write (summary)

            st.header ('Buy Recommendations')

            btdf = tdf [tdf.Action == 'Buy']            
            
            btdf [ 'clickable_url'  ] = btdf.apply(lambda row: "<a href='{}' target='_blank'>{}</a>".format(row.Source, "source link"), axis=1)
            btdf.rename(columns={'clickable_url':'Source Link'}, inplace=True)
            #st.write (btdf.columns.tolist())
            btdf[['Stock','Action','Reasons','Source Link']].to_html('/tmp/btdf.html',escape=False, index=False)
            
            with open('/tmp/btdf.html', 'r') as file:
                html_string = file.read()

            st.markdown(html_string, unsafe_allow_html=True)

            st.header ('Sell Recommendations')

            btdf = tdf [tdf.Action == 'Sell']            
            btdf [ 'clickable_url'  ] = btdf.apply(lambda row: "<a href='{}' target='_blank'>{}</a>".format(row.Source, "source link"), axis=1)
            btdf.rename(columns={'clickable_url':'Source Link'}, inplace=True)
            #st.write (btdf.columns.tolist())
            btdf[['Stock','Action','Reasons','Source Link']].to_html('/tmp/btdf.html',escape=False, index=False)
            
            with open('/tmp/btdf.html', 'r') as file:
                html_string = file.read()

            st.markdown(html_string, unsafe_allow_html=True)

            st.header ('Hold Recommendations')

            btdf = tdf [tdf.Action == 'Hold']            
            btdf [ 'clickable_url'  ] = btdf.apply(lambda row: "<a href='{}' target='_blank'>{}</a>".format(row.Source, "source link"), axis=1)
            btdf.rename(columns={'clickable_url':'Source Link'}, inplace=True)
            #st.write (btdf.columns.tolist())
            btdf[['Stock','Action','Reasons','Source Link']].to_html('/tmp/btdf.html',escape=False, index=False)
            
            with open('/tmp/btdf.html', 'r') as file:
                html_string = file.read()

            st.markdown(html_string, unsafe_allow_html=True)

            df = pd.read_csv ('https://investopsrecipes.s3.amazonaws.com/basic/all_stocks/just-all-custom-finviz.csv')
            
            btdf = tdf [tdf.Action == 'Buy']  
            stock_arr = btdf.Stock.unique().tolist()
            cols = ['Ticker', 'Company',  'Industry', 'Market Cap','Sales growth quarter over quarter', 'Profit Margin','Forward P/E', 'EPS growth this year','Performance (Week)', 'Performance (Month)','Relative Strength Index (14)', 'Analyst Recom', 'Relative Volume']
            print (df.columns)
            df = df [df.Ticker.isin (stock_arr)][cols]

            st.header ("Fundamental Analysis of Stocks with Buy Recommendations")
            st.dataframe(df)

        i = i + 1
    
    tabs.append ( 'StockNews' )

    with tabs [ -1 ] :
        df = pd.read_csv('https://investopsrecipes.s3.amazonaws.com/newsgpt/stock_recs.csv')
        st.dataframe(df)

streamlit_main ("https://worldopen.s3.amazonaws.com/eighth.csv")

