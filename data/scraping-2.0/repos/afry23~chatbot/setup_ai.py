from enum import Enum
from dotenv import load_dotenv
import openai

load_dotenv()
class LanguageModel(Enum):
    GPT4_TURBO = "gpt-4-1106-preview"
    GPT4 = "gpt-4"
    GPT3_TURBO = "gpt-3.5-turbo"
    DAVINCI = "text-davinci-003"
    CURIE = "text-curie-001"

# context = [ {'role':'system', 'content':"""
# You are a bot to help the user write code in Python.\
# If the user starts a general conversation, inform them about your purpose.\
# Structure your output as following, if the user requests a written code example:\
# Briefly, sum up what purpose the code is supposed to follow.\
# Then, provide the code solution.\
# Proceed to explain the code.\
# Finally, ask whether the user has any more questions left.\
# If not, inform the user that they should try the code and see whether it works.\
# Please inform the user to check the correct names of variables when giving a code example.\
# """} ]

context = []

def add_prompt(prompt):
    context = []
    context.append([ {'role':'system', 'content': prompt}])

def get_completion(prompt, model='gpt-3.5-turbo'):
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.Completion.create(
                engine=model,
                prompt=messages,
                max_tokens=150
            )
    return response.choices[0].message['content']

def get_chat_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0.5, stream=False):
    response = openai.ChatCompletion.create(
        model=model, # Genutztes Sprachmodell
        messages=messages,
        temperature=temperature, # Differenziertheit der Antwort
        stream=stream,
    )
    if stream:
        return response
    else:
        return response.choices[0].message["content"]

def collect_input(prompt):
    context.append({'role':'user', 'content':f"{prompt}"})

def collect_responses(response):
    context.append({'role':'assistant', 'content':f"{response}"})
