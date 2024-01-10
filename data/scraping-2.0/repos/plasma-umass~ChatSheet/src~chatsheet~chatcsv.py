import asyncio
import csv
import openai
import openai_async
import os

example_file = "test/examples/deniro.csv"

def precheck():
    if not 'OPENAI_API_KEY' in os.environ:
        print('You need a valid OpenAI key to use ChatDBG. You can get a key here: https://openai.com/api/')
        print('Set the environment variable OPENAI_API_KEY to your key value.')
        return False
    return True

async def chat(user_prompt):
    try:
        completion = await openai_async.chat_complete(openai.api_key, timeout=30, payload={'model': 'gpt-3.5-turbo', 'messages': [{'role': 'user', 'content': user_prompt}]})
        json_payload = completion.json()
        text = json_payload['choices'][0]['message']['content']
    except (openai.error.AuthenticationError, httpx.LocalProtocolError, KeyError):
        # Something went wrong.
        print()
        print('You need a valid OpenAI key to use ChatDBG. You can get a key here: https://openai.com/api/')
        print('Set the environment variable OPENAI_API_KEY to your key value.')
        import sys
        sys.exit(1)
    except Exception as e:
        print(f'EXCEPTION {e}, {type(e)}')
        pass
    print(text)
    # print(word_wrap_except_code_blocks(text))
    return text

    
def main():
    
    if not precheck():
        return

    with open(example_file, "r") as input_file:
        csvFile = csv.DictReader(input_file)
        l = list(csvFile)
        # print(l)

    user_prompt = f"Using the following data, answer this question: What films did DeNiro star in in the 1980s? {l}"

    asyncio.run(chat(user_prompt))

    return

    
