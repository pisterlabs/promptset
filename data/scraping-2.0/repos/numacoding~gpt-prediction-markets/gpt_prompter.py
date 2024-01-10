import openai
import os
from dotenv import load_dotenv, find_dotenv
from datetime import datetime, timedelta
_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')

max_tokens = 4097

def get_prediction_markets(linklist, model="gpt-3.5-turbo"):

    end_date = str(datetime.now() + timedelta(days=2))
    print(end_date)

    prompt = f'''
        I will need you to take the text that I will provide you (at the end of this message, \
        delimited by triple backticks), process their content, filter the text that grammatically makes sense and based on this, prepare questions for prediction \
        markets. The information that you provide about every market \
        must be structured in JSON format. If you generate more than one Prediction Market, the result should be a list of JSON files. \
        These JSON files must contain the following keys (in addition, I will provide specifications for each one): \
        - question: statement detailing the event \
        - type of market: specify if the market is binary, categorical or scalar \
        - tokens: if the market is binary or categorical, it specifies all the possible scenarios of the event. \
        If it is scalar, it provides a minimum and maximum value according to the news in question \
        - End Date: this date must be prior to the occurrence of the event, specified in YYYY-MM-DDTHH:MM format \
        in UTC time (for example, if an event occurs on April 12, 2023 at 3:30 p.m. in UTC time, the closing \
        date must be before 2023-04-12T15:30:00) \
        - Note: a brief description of the market should go here, giving context of the event covered, the meaning \
        of each of the tokens (possible outcomes), and a data source that can be used to find this information. \
        Ideally, this source should be a publicly accessible database, API, or Python code. \
        Check our rules here: https://docs.zeitgeist.pm/docs/learn/market-rules \

        IMPORTANT: DO NOT MENTION PREDICTION MARKETS WHOSE COMPLETION DATE IS BEFORE THIS DATE: {end_date} \

        ```{linklist}```
    '''
    
    #Check token limit
    #1000 tokens are equal to 750 words (more or less)
    max_words = int(round(max_tokens*0.75,0))

    # Check if the number of tokens exceeds the maximum allowed
    if len(prompt.split()) >= max_words:
        #find the position of the previous '.' before the limit
        position = prompt.rfind(".", 0, max_words)
        prompt = prompt[:position]

    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, #degree of randomness of the model's output
    )
    return response.choices[0].message["content"] # type: ignore
