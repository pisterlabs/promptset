import snowflake.connector
import pandas as pd 
import openai
import os 
import get_date
from datetime import datetime, timedelta
import tiktoken

def count_and_limit_tokens(string: str, encoding_name: str, encoding_model:str, limit_num:int) -> str:
    """Returns the number of tokens in a text string.
    
    Args: 
        string (str): string to process 
        encoding_name (str): encoding name to put into tiktoken 
        encoding_model (str) : encoding model to put into tiktoken 
        limit_num (int) : maximum number of tokens to be extracted 
    returns: 
        str: new string that was processed after limiting the number of tokens """
    enc = tiktoken.get_encoding(encoding_name)
    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    enc = tiktoken.encoding_for_model(encoding_model)
    # Encode the string 
    tokens = enc.encode(string)
    # Get the number of tokens and limit it if it's bigger than maximum number of tokens
    num_tokens = len(tokens)
    max_tokens = min(num_tokens, limit_num)
    # truncate the text accordingly 
    limited_text = enc.decode(tokens[:max_tokens])
    
    return limited_text



def summarize_comment(channel_name:str, query_date=datetime.now())-> str: 
    """
    gets chats ranging between query_date and query_date+5minutes and summarizes it into 2 sentences
    channel_name: name of the streamer to be analyzed (passed on from the airflow function)
    query_date: the starting date of the chats
    """
    to_date = query_date.strftime("%Y-%m-%d %H:%M")
    from_date = (query_date - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M")

    #set up snowflake connecection 
    conn = snowflake.connector.connect(
        user = os.environ["SNOWFLAKE_USER"], 
        password = os.environ["SNOWFLAKE_PW"],
        account = 'MNB68659.us-west-2',
        warehouse = 'compute_wh',
        database = 'stream_data_anal',
        schema = 'twitch_data')

    # Read target data - data from target streamer with today's stream where the topic 
    query = f"SELECT * from twitch_chats WHERE CHANNEL_NAME = '{channel_name}'  AND MESSAGE_AUTHOR NOT LIKE '%bot' AND MESSAGE_DATE BETWEEN '{from_date}' AND '{to_date}'"
    df = pd.read_sql_query(query, conn)

    openai.api_key = os.environ["OPENAI_KEY"]
    encoding_name = "p50k_base"
    encoding_model = "text-davinci-002"
    # Check the number of tokens in the text and truncate it if it's longer than 4000 tokens 
    message = '. '.join(i for i in df['MESSAGE_TEXT'])
    message = count_and_limit_tokens(string=message, encoding_model=encoding_model, encoding_name=encoding_name, limit_num=4000)
    
    # set up the ChatGPT API endpoint and parameters
    url = 'https://api.openai.com/v1/engines/text-curie-001/completions'

    # Query the OPENAI API 
    response = openai.Completion.create(
        engine=encoding_model,
        prompt=f"These are the chats from a twitch stream. Please write me a two sentence summary of what the viewers are saying in this sentence: {message}",
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
        seed = 42
    )

    try: 
        # Extract the summary info 
        summary = response['choices'][0]['text'].replace('\n', '')
        # Create data to insert into the table 
        data = (channel_name, from_date, summary)
        cur = conn.cursor() 

        # Insert the data into database 
        try_cnt = 0 
        while try_cnt <= 3: 
            try: 
                cur.execute(f"INSERT INTO twitch_chat_summaries (summary_channel,summary_date,summary_content)\
                            VALUES (%s,%s,%s)", data)
                conn.commit()
                cur.close()
                conn.close()
                break

            # If it fails try 2 more times 
            except Exception as e: 
                print(e)
                try_cnt += 1 
        print(f"inserted summary data from {from_date} to {to_date} to Snowflake")
    except Exception as e: 
        print(f"Failed to insert data from {from_date} to {to_date}")
    
    
    return channel_name 

