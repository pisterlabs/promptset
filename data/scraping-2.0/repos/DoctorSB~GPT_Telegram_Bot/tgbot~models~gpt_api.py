import openai
from environs import Env
import backoff
import time

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=30)
def completions_with_backoff(**kwargs):
    try:
        return openai.Completion.create(**kwargs)
    except Exception as e:
        print(f"Error details: {str(e)}")
        return None

class GPT:
    def __init__(self) -> None:
        env = Env()
        env.read_env()
        openai.api_key = env('GPT_API_KEY')
        self.__message = []

    def request(self, task):
        self.__message.append({'role': 'user', 'content': task})
        print(f'{task}: запрос отправлен')
        try:
            start_time = time.time()
            answer = completions_with_backoff(model="gpt-3.5-turbo", prompt=task)
            end_time = time.time()
            print(f'Request completed in {end_time - start_time} seconds')
            if answer:
                self.__message.append(
                    {'role': 'assistant', 'content': answer.choices[0].text})
                return answer.choices[0].text
            else:
                print("No answer received.")
                return None
        except Exception as e:
            print(f"Error encountered: {str(e)}")
            return None

    def clear(self):
        self.__message = []

if __name__ == '__main__':
    gpt = GPT()
    data = input('Введите задачу: ')
    print(gpt.request(data))
