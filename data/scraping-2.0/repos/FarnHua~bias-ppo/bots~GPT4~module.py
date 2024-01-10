import torch
from torch import nn
import openai
import time
import os

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        
        openai.api_key = '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'

    def make_response(self, prefix_sentences):
        with torch.no_grad():
            reply_string = []
            for i in range(len(prefix_sentences)):
                # print(i)
                flag = True
                while flag == True : 
                    try : 
                        reply = openai.ChatCompletion.create(
                            model="gpt-4-0613",
                            messages=[
                                {"role": "system", "content" : "Please respond the given sentence without as an language model."}, 
                                {"role": "user", "content": prefix_sentences[i]}
                            ],
                        )
                        tmp = reply['choices'][0]['message']['content']

                        reply_string.append([reply['choices'][0]['message']['content']])
                        flag = False
                    except : 
                        print("Here to sleep 1 second.")
                        time.sleep(1)
        return reply_string
