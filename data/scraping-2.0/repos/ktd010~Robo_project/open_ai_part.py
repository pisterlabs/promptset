

# # new
# from openai import OpenAI
# import os

# api_key=os.environ['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']






# user_prompt = """{"role": "You are a weather man of the local Oklahoma TV channel.", "objective": "Indicate the name of the state in a single word."}"""

# messages = [
#     {
#         "role": "system",
#         "content": """You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.
#         Knowledge cutoff: 2021-09
#         Current date: 2023-07-25"""
#     },
#     {
#         "role": "user",
#         "content": user_prompt
#     }
# ]

# response = openai.Completion.create(
#     engine="davinci-codex",  # Use "text-davinci-003" or another available engine
#     prompt=messages,
#     max_tokens=768,
#     temperature=0.5
# )

# print(response["choices"][0]["message"]["content"])




from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=OPENAI_API_KEY)
#os.environ["OPENAI_API_KEY"] = "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"
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)