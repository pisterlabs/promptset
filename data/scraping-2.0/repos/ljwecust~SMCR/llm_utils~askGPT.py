import time
import openai
import copy
from openai.error import RateLimitError


def askGPT_meta(messages):

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages,
        temperature=0,
        seed=42,
    )
    return completion.choices[0].message.content


class ASK_GPT(object):
    def __init__(self, key_list: list, time_sleep=0, error_sleep=20, num_retry=10):
        self.time_sleep = time_sleep
        self.error_sleep = error_sleep
        self.num_retry = num_retry

        openai.api_key = key_list[0]
    
    def askGPT4Use_nround(self, messages):
        res = 'None'; status_code = 0

        for _ in range(self.num_retry):
            status_code += 1
            
            try:
                time.sleep(self.time_sleep)
                res = askGPT_meta(messages)
                status_code -= 1
                break
            except RateLimitError as e:
                print(e)
                time.sleep(self.error_sleep)
            except Exception as e:
                print(e)
                time.sleep(self.error_sleep)

        if status_code == self.num_retry:
            print('#'*42 + 'f a i l')
        return res

