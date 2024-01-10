import os
import openai
from config import OPENAI_KEY

from upper import open_txt

openai.api_key = OPENAI_KEY

def simple_inference(message):
        result = openai.Completion.create(
                engine="text-davinci-002", #davinci", # "curiedavinci",
                prompt=message,
                temperature=0.9,
                max_tokens=200,
                top_p=1,
                frequency_penalty=0.3,
                presence_penalty=0.1,
                stop=["\""]
        )
        return result['choices'][0]['text']


def simple_inference_fine_tuned(message):
        result = openai.Completion.create(
                model="davinci:ft-personal-2022-03-02-01-11-13", #5 epoch
                # model="davinci:ft-personal-2022-03-01-21-16-08", # 3 epoch
                prompt=message,
                temperature=0.9,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0.6,
                presence_penalty=0.3,
                stop=["Ivan"]
        )
        return result['choices'][0]['text']



if __name__ == "__main__":
        # print(simple_inference_fine_tuned(open_txt('/home/vova/myprojects/nlp/text.txt')))
        print(simple_inference(open_txt('/home/vova/myprojects/nlp/text.txt')))
        
