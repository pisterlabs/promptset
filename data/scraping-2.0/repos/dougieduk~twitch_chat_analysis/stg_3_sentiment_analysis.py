import snowflake.connector
import pandas as pd 
import openai
import os 
import get_date




def sentiment_analysis(channel_name:str, yy_mm:str=get_date.get_four_digit_date())-> str: 
    """
    analyzes the sentiment of each comment and get the ratio of joyful / melancholic/ anxious/furious/ grateful/ mixed
    channel_name: name of the streamer to be analyzed (passed on from the airflow function)
    """
    df_new = pd.DataFrame(columns = ['CHANNEL_NAME', 'CHANNEL_DATE', 'MESSAGE_DATE', 'MESSAGE_AUTHOR','MESSAGE_TEXT', 'joyful_sentiments','melancholic_sentiments', 'anxious_sentiments','furious_sentiments','grateful_sentiments','mixed_sentiments'])

        
    #set up snowflake connecection 
    conn = snowflake.connector.connect(
        user = os.environ["SNOWFLAKE_USER"], 
        password = os.environ["SNOWFLAKE_PW"],
        account = 'MNB68659.us-west-2',
        warehouse = 'compute_wh',
        database = 'stream_data_anal',
        schema = 'twitch_data')

    # Read target data - data from target streamer with today's stream where the topic 
    query = f"SELECT * from twitch_chats WHERE CHANNEL_NAME = '{channel_name}' AND CHANNEL_DATE = '{yy_mm}' AND SENTI_PROCESSED = FALSE AND message_author NOT LIKE '%bot'"
    df = pd.read_sql_query(query, conn)

    # set up the ChatGPT API endpoint and parameters
    url = 'https://api.openai.com/v1/engines/text-curie-001/completions'
    headers = {
        'Content-Type': 'application/json'
    }
    params = {
        'prompt': 'Perform Sentiment Analysis',
        'max_tokens': 1,
        'temperature': 0.7,
        'n': 1,
        'stop': ['\n']
    }

    openai.api_key = os.environ["OPENAI_KEY"]
    cur = conn.cursor() 

    # Run the messages through OPEN AI API
    for ind, message in enumerate(df['MESSAGE_TEXT']):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Sentiment analysis with probabilities in decimal form for joyful, melancholic, anxious, furious, grateful, and mixed. Make sure to output the message in the following format: Joyful: corresponding score \n Melancholic: corresponding score \n Anxious: corresponding score \n Furious: corresponding score \n Grateful: corresponding score \n Mixed: corresponding score \n for this message: {message} \n Make sure the probabilities add to 1!",
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
            seed = 42
        )
        # Extract the scores 
        print(f"processing sentiment: {ind}")
        sentiment_score = response['choices'][0]['text']
        scores = sentiment_score.strip().split('\n')
        scores = [s.strip() for s in scores if s.strip()]
        try: 
            # Assign scores to different variables 
            joyful = float(scores[0].split(':')[1].strip())
            melancholic = float(scores[1].split(':')[1].strip())
            anxious = float(scores[2].split(':')[1].strip())
            furious = float(scores[3].split(':')[1].strip())
            grateful = float(scores[4].split(':')[1].strip())
            mixed = float(scores[5].split(':')[1].strip())

            # Update the DataFrame 
            df.loc[ind,'JOYFUL_SENTIMENTS'] = joyful
            df.loc[ind,'MELANCHOLIC_SENTIMENTS'] = melancholic
            df.loc[ind,'ANXIOUS_SENTIMENTS'] = anxious
            df.loc[ind,'FURIOUS_SENTIMENTS'] = furious
            df.loc[ind,'GRATEFUL_SENTIMENTS'] = grateful
            df.loc[ind,'MIXED_SENTIMENTS'] = mixed

            # Update  the data 
            try_cnt = 0 
            while try_cnt <= 3: 
                try: 
                    cur.execute(f"UPDATE twitch_chats SET joyful_sentiments='{df.loc[ind,'JOYFUL_SENTIMENTS']}', \
                                melancholic_sentiments='{df.loc[ind,'MELANCHOLIC_SENTIMENTS']}',\
                                anxious_sentiments='{df.loc[ind,'ANXIOUS_SENTIMENTS']}',\
                                furious_sentiments='{df.loc[ind,'FURIOUS_SENTIMENTS']}',\
                                grateful_sentiments='{df.loc[ind,'GRATEFUL_SENTIMENTS']}',\
                                mixed_sentiments='{df.loc[ind,'MIXED_SENTIMENTS']}',\
                                senti_processed=TRUE\
                                WHERE chat_id={df.loc[ind,'CHAT_ID']}")
                    break
                # If the update fails try again two more times 
                except Exception as e: 
                    print(e)
                    try_cnt += 1 
            print(f"inserted data to Snowflake: {ind}")
    

        except Exception as e:
            print(f'Unable to extract sentiment {ind}')
            print(e)


