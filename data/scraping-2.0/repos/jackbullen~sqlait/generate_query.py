import openai
import dotenv
import os

dotenv.load_dotenv()
key = os.getenv("OPENAI_API_KEY")

def generate_sql_query(messages):
    response = openai.ChatCompletion.create(
      api_key=key,
      model='gpt-4',
      messages = messages,
      max_tokens=2048
    )
    messages = response["choices"][0]["message"]
    sql_query = response["choices"][0]["message"]["content"].split('```')[1][4:]
    return sql_query, messages