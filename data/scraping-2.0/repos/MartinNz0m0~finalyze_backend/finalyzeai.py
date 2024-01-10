import pandas
import os
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from dotenv import load_dotenv
import mpesa_funcs

load_dotenv()

def finalyze_ai_chat(user, message):
    try:
        df = mpesa_funcs.mpesa_df(f'mpesa_{user}')
        llm = OpenAI(api_token=os.getenv('OPEN_API'))

        df = SmartDataframe(df, config={
            'llm': llm,
            'enable_cache': False,
            "conversational": True,
            "enforce_privacy": True,
        })
        response = df.chat(message)
        return response
    except Exception as e:
        print(e)
        return "Sorry, I didn't get that. Please try again."

