# import hydra
import numpy as np
import json
import logging 
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time 
from collections import defaultdict
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# set logging level to info
logging.basicConfig(level=logging.INFO)

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()
    
def load_tensorboard_logs(path):
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)
    
    return data

class LLMLunarEnvGenerator:
    
    def __init__(self, cfg: dict):
                #  horizon: int, 
                #  top, 
                #  bottom, 
                #  model: str = "gpt-3.5-turbo", 
                #  temperature: float = 0.5,
                #  sample: int = 4,
                #  smooth_window: int = 1
                 
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("LLMLunarEnvGenerator initializing...")
        # self.horizon = cfg['horizon']
        # self.top = cfg['top'] +0.01
        # self.bottom = cfg['bottom']-0.01
        self.model = cfg['model'] 
        self.temperature = cfg['temperature']
        self.sample = cfg['sample']
        self.chunk_size = 4
        # self.smooth_window = cfg['smooth_window']
        
        openai.api_key = cfg['openaikey']
        logging.info(f"OpenAI API Key: {openai.api_key}")
        # This is to store all the responses.
        self.responses = []

        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime('%d-%m-%Y-%H-%M-%S')
       
        if not os.path.exists('./llmlog'):
            os.makedirs('./llmlog')
        self.message_log = f"llmlog/message_{formatted_date}.txt"
        logging.info(f"Message is logged in: {self.message_log}")

    
       # Loading all text prompts
        prompt_dir = './TeachMyAgent_modified/LLM_lunar/Prompt'
        self.initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
        
        self.code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
        self.code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
        self.initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
        # reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
        self.policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
        self.execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
        # terrain_example = file_to_string(f'{prompt_dir}/terrain_example.txt')
        
        
        self.initial_system += self.code_output_tip 
        # self.initial_system += terrain_example
        
        self.messages = [{"role": "system", "content": self.initial_system}, {"role": "user", "content": self.initial_user}]

        self.logger.info("LLMLunarEnvGenerator initialized")

    def _log_messge(self,message,log_format='json'):
        logging.info(f"Logging Message...")
        if log_format == 'str':
            assert isinstance(message,str), "message must be a string!"
            with open(self.message_log, 'a') as file:
                file.write(message)
                file.write('\n')
                
        elif log_format == 'json':
            assert isinstance(message,dict), "message must be a dict!"
            with open(self.message_log, 'a') as file:
                json.dump(message, file, indent=4)
        else:
            raise NotImplementedError(f"log_format {log_format} not implemented!")
        
    def init_generate(self,debug:bool = False):
        logging.info("Generating initial terrain...")
        message = self._callOpenAI(self.messages,debug,init=True)
        return message

    def iter_generate(self,tensorboard_logdir: str = None,debug:bool = False):
        logging.info("Generating terrain with updated metric...")
        self._update_message(tensorboard_logdir)
        message = self._callOpenAI(self.messages,debug) 
        return message

    def _callOpenAI(self,messages,debug:bool = False,init=False):
    
        response_cur = None
        logging.info(f"Generating samples with {self.model}")
        total_samples = 0
        logging.debug(f"Messages: {messages}\n*********************")
        if not init:
            while True:
                for attempt in range(10):
                    try:
                        response_cur = openai.ChatCompletion.create(
                            model=self.model,
                            messages=messages,
                            # temperature=self.temperature,
                            # n=self.chunk_size
                        )
                        total_samples += self.chunk_size
                        logging.info("LLM call succeeded!")
                        break
                    except Exception as e:
                        if attempt >= 10:
                            self.chunk_size = max(int(self.chunk_size / 2), 1)
                        logging.info(f"Attempt {attempt+1} failed with error: {e}")
                        time.sleep(1)

                if response_cur is None:
                    logging.info("Code terminated due to too many failed attempts!")
                    exit()

                
                
                raw_ret = response_cur['choices'][0]['message']['content']
                try:
                    extract = raw_ret.split('{')[-1].split('}')[0].split(',')
                    ret_key = [x.split(':')[0] for x in extract]
                    ret_value = [float(x.split(':')[1]) for x in extract]
                    ret = {}
                    for i in range(3):
                        ret[eval(ret_key[i])] = ret_value[i]
                    self.logger.info(f"***********************\nRet: {ret}\n************************")
                    break 
                except Exception as e:
                    logging.info(f"Error: {e}")
                    logging.info(f"Ret: {ret}")
                    logging.info("Ret is not a valid python expression, trying again...")
                    continue    
        else:
            ret = {'gravity': -10.0, 'wind_power': 0.0, 'turbulence_power': 0.0}
            response_cur = {}
            response_cur['choices'] = [{'message': {'content': str(ret)}}]
            
        self._log_messge(response_cur)
        self.responses.extend(response_cur['choices'])

        logging.info(f"This is the generated settings: {ret}")
        return ret
    
    
    def _update_message(self,tensorboard_logdir: str = None):

        assert tensorboard_logdir is not None, "tensorboard_logdir must be provided!"

        # code_feedbacks = []
        # contents = []
        # successes = []
        # reward_correlations = []
        content = ''
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        logging.info(f"Reading tensorboard logs from {tensorboard_logdir}")
        max_iterations = np.array(tensorboard_logs['critic_loss']).shape[0]
        epoch_freq = max(int(max_iterations // 10), 1)

        content += self.policy_feedback.format(epoch_freq=epoch_freq)
        logging.info(f"Epoch Frequency: {epoch_freq}")
        logging.info(f"content: {content}")
        logging.info(tensorboard_logs.keys())
        # Add reward components log to the feedback
        for metric in tensorboard_logs:
            if "/" not in metric and len(tensorboard_logs[metric]) > 0:
                metric_cur = [round(x,2) for x in tensorboard_logs[metric][::epoch_freq]]
                # print(metric)
                # print(tensorboard_logs[metric])
                metric_cur_min = min(tensorboard_logs[metric])
                metric_cur_max = max(tensorboard_logs[metric])
                metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])           
                metric_name = metric
                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    

        # code_feedbacks.append(self.code_feedback)
        content += self.code_feedback  
        content += self.code_output_tip
        # logging.info(f"context: {content}")
        if len(self.messages) == 2:
            self.messages += [{"role": "assistant", "content": self.responses[0]["message"]["content"]}]
            self.messages += [{"role": "user", "content": content}]
        else:
            assert len(self.messages) == 4
            self.messages[-2] = {"role": "assistant", "content": self.responses[0]["message"]["content"]}
            self.messages[-1] = {"role": "user", "content": content}

        logging.info("Message Updated with the newest metric!")

