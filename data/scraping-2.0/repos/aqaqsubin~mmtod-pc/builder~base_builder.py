import time
import json
import random
import openai
import numpy as np

from typing import Optional, Tuple, Dict, List
from os.path import join as pjoin
from openai.error import RateLimitError, APIError, APIConnectionError

from data_utils.logger import Logger

class BaseDatasetBuilder:
    def __init__(self, args, leftovers):
        self.setup_seeds(args.seed)

        self.data_dir = args.data_dir
        self.image_dir = pjoin(self.data_dir, 'images')
        self.annot_dir = pjoin(self.data_dir, 'annotations')
        self.prompt_dir = args.prompt_dir
    
        self.logger = Logger(logger_name=args.logger_name, dirpath=args.log_path)
        
        self.emotion_kor_to_eng = {
            "기쁨" : "happy",
            "당황" : "embarrassing",
            "분노" : "angry",
            "불안" : "unrest",
            "상처" : "hurt",
            "슬픔" : "sad",
            "중립" : "neutral",
        }
        self.drop_emotions = []
        self.system_message = None
        self.fewshot_samples = None

        with open(args.config_file, 'r', encoding='utf-8') as df:
            openai.api_key = json.load(df)['api-key']
    
    def setup_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def preprocess_dialogue_fewshot(fewshot_samples: List[Dict]) -> List[Dict]:
        pass

    def _get_gpt_prompt(self, age, sex, emotion) -> str:
        pass

    def _get_gpt_prompt_example(self, annotation):
        pass

    def _get_gpt_response(self, prompt: str, num_fewshot: int):
        try:
            messages=[{"role": "system", "content": self.system_message}]
            
            samples = self.fewshot_samples if num_fewshot >= len(self.fewshot_samples) \
                else random.choices(self.fewshot_samples, k=num_fewshot)
            for sample in samples:
                messages.append({"role" : "user", "content": sample['context']})
                messages.append({"role" : "assistant", "content": '\n'.join(sample['response'])})
            messages.append({"role" : "user", "content": prompt})
            
            reply = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            output = reply['choices'][0]['message']['content']
            return output
        
        except RateLimitError as rate_e:
            self.logger.error(f"RateLimitError: Delay 1 minute ({prompt})")
            time.sleep(60 * 1)
            return self._get_gpt_response(prompt, num_fewshot)
        
        except APIError as bad_gateway_e:
            self.logger.error(f"APIError: Delay 1 hour ({prompt})")
            time.sleep(60 * 30)
            return self._get_gpt_response(prompt, num_fewshot)
        
        except APIConnectionError as conn_e:
            self.logger.error(f"APIConnectionError: Delay 1 hour ({prompt})")
            time.sleep(60 * 60)
            return self._get_gpt_response(prompt, num_fewshot)
        
        except Exception as unk_e:
            if "maximum context length" in f"{unk_e}":
                self.logger.error(f"{unk_e}: Delay 1 minute ({prompt})")
                time.sleep(60 * 1)
                return self._get_gpt_response(prompt, num_fewshot=num_fewshot - 2)
            
            elif "Request timed out" in f"{unk_e}":
                self.logger.error(f"{unk_e}: Delay 5 minute ({prompt})")
                time.sleep(60 * 5)
                return self._get_gpt_response(prompt, num_fewshot=num_fewshot)
            
            elif "overloaded or not ready yet" in f"{unk_e}":
                self.logger.error(f"{unk_e}: Delay 10 minute ({prompt})")
                time.sleep(60 * 10)
                return self._get_gpt_response(prompt, num_fewshot=num_fewshot)
            
            self.logger.error(f"{unk_e}: Delay 1 hour ({prompt})")
            time.sleep(60 * 60)
            return self._get_gpt_response(prompt, num_fewshot)
