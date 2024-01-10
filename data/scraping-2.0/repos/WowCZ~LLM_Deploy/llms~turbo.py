import random
import openai
import time
from llms import LLMAPI, get_logger
from typing import List
from pydantic import BaseModel

logger = get_logger(__name__, 'INFO')

with open('assets/openai_keys.txt', 'r') as fr:
    keys = fr.readlines()
    ks = []
    for k in keys:
        ks.append(k.strip())

key_id = random.randint(0, len(ks)-1)
openai_key = ks[key_id]

class TurboAPI(LLMAPI):
    def __init__(self, 
                 model_name='gpt-3.5-turbo', 
                 model_path=None,
                 model_version='default'):
        super(TurboAPI, self).__init__(model_name, model_path, model_version)
        self.name = 'turbo'
        
    def generate(self, item: BaseModel) -> List[str]:
        openai.api_key = openai_key

        prompt = item.prompt
        
        if not prompt:
            return []

        if type(prompt) is not list:
            prompt = [prompt]

        error = None
        num_failures = 0
        while num_failures < 5:
            try:
                turbo_replies = []
                for p in prompt:
                    chat_instance = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}]

                    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                                messages=chat_instance,
                                                                temperature=item.temperature,
                                                                top_p=item.top_p,
                                                                max_tokens=item.max_new_tokens,
                                                                n=item.num_return)
                    turbo_reply = completion.choices[0].message.content.strip()
                    turbo_replies.append(turbo_reply)
                
                return turbo_replies
            except openai.error.RateLimitError as error:
                logger.warning(error)
                logger.warning(f"Reach Rate Limit!")
                num_failures += 1
                time.sleep(5)
            except Exception as error:
                logger.warning(error)
                num_failures += 1
                time.sleep(5)

        raise error
    
if __name__ == '__main__':
    model_api = TurboAPI()
