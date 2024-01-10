import openai
import time
openai.api_key = 'api key'

def get_completion(messages, model="gpt-3.5-turbo", max_tokens=100,temperature=0):
    response = ''
    except_waiting_time = 0.1
    while response == '':
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                request_timeout=50,
                max_tokens= max_tokens
            )
        except Exception as e:
            time.sleep(except_waiting_time)
            if except_waiting_time < 2:
                except_waiting_time *= 2
    return response.choices[0].message["content"]
