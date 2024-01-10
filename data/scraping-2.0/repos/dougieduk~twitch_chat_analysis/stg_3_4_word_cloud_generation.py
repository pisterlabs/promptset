import snowflake.connector
import pandas as pd 
import openai
import os 
import get_date
import save_to_snowflake
from datetime import *
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


def count_words(query_date:datetime, channel_name:str, yy_mm:str=get_date.get_four_digit_date()) -> str:
    """
    Args: 
        query_date(datetime): the date when the analysis will be run\n
        channel_name(string): the channel name/streamer name \n
        yy_mm(string): four digit of current date (2024/04/24->0424)
    
    Returns: 
        channel_name(string): the channel name/streamer name to be passed onto the next function \n
    
    """
    # Create connection 
    #set up snowflake connecection 
    SNOWFLAKE_USER = os.environ["SNOWFLAKE_USER"]
    SNOWFLAKE_PW = os.environ["SNOWFLAKE_PW"]
    conn = snowflake.connector.connect(
        user = SNOWFLAKE_USER, 
        password = SNOWFLAKE_PW,
        account = 'MNB68659.us-west-2',
        warehouse = 'compute_wh',
        database = 'stream_data_anal',
        schema = 'twitch_data')
    
    # Read data for last five minutes 
    to_date = query_date.strftime("%Y-%m-%d %H:%M")
    from_date = (query_date - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M")
    
    # Get the chat data 
    query = f"SELECT * from twitch_chats WHERE CHANNEL_NAME = '{channel_name}' AND CHANNEL_DATE = '{yy_mm}' AND message_date BETWEEN '{from_date}' and '{to_date}'"
    df = pd.read_sql_query(query, conn)
    
    # Get emoji texts
    emoji_query = f"SELECT NAME FROM emojis WHERE channel_name = '{channel_name}' OR channel_name = 'global'"
    emoji_df = pd.read_sql_query(emoji_query, conn)

    # Get a set of emojis so we could exclude the words
    emoji_set = set(emoji_df["NAME"].str.lower())
    
    # Concat the messages for comparison 
    all_messages = ' '.join(df["MESSAGE_TEXT"].tolist())

    # Tokenize the combined string into a list of words
    words = word_tokenize(all_messages)
    
    # Remove stopwords and non-alphabetic characters
    stop_words = set(stopwords.words('english'))
    # Exclude stop_words and emojis 
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words and word.lower() not in emoji_set]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='n') for word in words]

    # Get the part-of-speech tags for the words
    nouns = [word for word in words if wordnet.synsets(word) and word.lower()!='medium']

    # Create a word count dict 
    word_count = dict()
    for noun in nouns:
        word_count[noun] = word_count.get(noun, 0) + 1
    
    # Create a word count df 
    word_count_df = pd.DataFrame(dict(word_count), index=["count"]).T
    word_count_df = word_count_df.sort_values('count', ascending=False)
    word_count_df["channel_name"] = channel_name 
    word_count_df["channel_date"] = yy_mm
    word_count_df["count_date"] = from_date
    word_count_df["count_date"] = pd.to_datetime(word_count_df["count_date"])
    # Reset index and rename it
    word_count_df = word_count_df.reset_index()
    word_count_df = word_count_df.rename(columns={"index":"word"}) 
    # Remove other words 
    word_count_df = word_count_df.query("word != 'http'")
    # Get the top 150 results 
    concat_len = min(150, len(word_count_df))
    word_count_df = word_count_df.iloc[:concat_len]
    
    # Save to DB 
    dbengine = save_to_snowflake.set_up_engine(snowflake_schema='twitch_data',snowflake_database='stream_data_anal',snowflake_user=SNOWFLAKE_USER,snowflake_password=SNOWFLAKE_PW, snowflake_account='MNB68659.us-west-2',snowflake_wh='compute_wh', snowflake_role="accountadmin")
    word_count_df.to_sql(if_exists='append', name="word_counts", con=dbengine, index=False)

    return channel_name 
    
    

    