import openai
from openai import OpenAI
import time
import traceback
import os
client = OpenAI(
    # This is the default and can be omitted
    api_key="YOUR-API-KEY",
    organization="YOUR-ORGANIZATION"
)
def askGpt(prompt: str,model:str): # this returns gpt3s response
    completion = None
    TOLERANCE = 10
    tolerance_counter = 0
    while completion==None and tolerance_counter < TOLERANCE:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5)
            print(completion)
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
            tolerance_counter+=1
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            time.sleep(5)
            tolerance_counter+=1
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
            tolerance_counter+=1
        except: # a failsafe in case there is a problem with the API being overloadeds
            time.sleep(5)
            traceback.print_exc()
            tolerance_counter+=1
    return completion.choices[0].message["content"]
