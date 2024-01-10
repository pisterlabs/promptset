from utils.api_key import gpt_api_key
from openai import OpenAI


def generate_sql_query(nlp_prompt):
    client = OpenAI(api_key=gpt_api_key)

    chat_completion = client.chat.completions.create(
        messages=[ {"role": "system", "content":"I am expert in generating sql query from natural language. Assume default names and schema if specific information is not provided. Return only the query"},
                   {'role': 'user', 'content': nlp_prompt} ],
        model="gpt-3.5-turbo"
    )
    
    return chat_completion.choices[0].message.content.strip()