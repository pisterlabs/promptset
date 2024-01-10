# import modules
from dotenv import load_dotenv
import os
from openai import OpenAI

# env variables, constants and API keys
load_dotenv()
openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKey_BACKUP')
serpapi_api_key=os.getenv("SERPAPI_API_KEY", "YourAPIKey_Backup")
# print(serpapi_api_key) # check
# print(openai_api_key)

client = OpenAI()


"""
/// QUICKSTART ///
"""

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
print(response)
"""
ChatCompletion(id='chatcmpl-8dI6qmgl3tR0iBo5cn9xyiCDsNqgD', 
choices=[Choice(finish_reason='stop', 
index=0, message=ChatCompletionMessage(content='The 2020 World Series was played at Globe Life Field in Arlington, Texas.', role='assistant', function_call=None, tool_calls=None), logprobs=None)], 
created=1704375096, model='gpt-3.5-turbo-0613', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=17, prompt_tokens=53, total_tokens=70))

Next up - study chat completion object / quickstart guide below:
 |      Returns:
 |          A dictionary representation of the model.
"""
print(response.choices[0].message)
"""
ChatCompletionMessage(content='The World Series in 2020 was played in Arlington, Texas at the Globe Life Field.',
role='assistant', function_call=None, tool_calls=None)
"""
# help(response)
print(response.dict()['choices'][0]['message']['content'])
"""
The World Series in 2020 was played at Globe Life Field in Arlington, Texas.
"""


# help(response)
"""
https://platform.openai.com/docs/api-reference/chat
-> goes into detail on chat completion, chat completion objects, and chat completion chunk objects.
"""



















"""
/// WALKTRHOUGH - https://platform.openai.com/docs/guides/text-generation ///
"""
