import json
import openai
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import os
import torch

from collections import OrderedDict


class FunnyUtteranceFiltering:
    """ Main usecase """

    def __init__(self, lm, args):

        vids = torch.load(args.video_ids)
        with open(args.video_info) as json_file:
            info = json.load(json_file)
            
        self.data = dict([(k, v) for k, v in list(info.items()) if k in vids])
        self.prompt_result_path = args.prompt_result_path
        self.df = pd.DataFrame(dict(
            id=[],
            dv_funny_utterance=[],
            d_funny_utterance=[],
            dv_explanation=[],
            d_explanation=[],
            vcap=[],
            stt=[],
            
        ))
        self.lm = lm
        
        # remove repeatition
        if os.path.exists(self.prompt_result_path):
            with open(self.prompt_result_path) as f:
                exist_json = json.load(f)
            for k, v in exist_json.items():
                del self.data[v['id']]
                self._save_result(
                    v['id'],
                    v['dv_funny_utterance'], v['d_funny_utterance'],
                    v['dv_explanation'], v['d_explanation'],
                    v['vcap'], v['stt']
                )
        

    def __call__(self):
        for key, val in tqdm(self.data.items()):
            if type(val['vcap']) == int:
                continue
            description = ' '.join(val['vcap'].split('/')[2:])
            utters = val['stt']['text']

            dv_respondent = VisualDialogueRespondent(self.lm, description, utters)
            d_respondent = DialogueRespondent(self.lm, utters)

            # Get dv funny utterance
            dv_funny_utterance = dv_respondent.find_funny_utterance()
            if 'no.' in dv_funny_utterance.lower()[:5]:
                self._save_result(key, 'No.', None, None, None, val['vcap'], val['stt']['text']) # first filter : no verbal effect
                continue
            
            # Get d funny utterance
            d_funny_utterance = d_respondent.find_funny_utterance()
            if 'no.' in d_funny_utterance.lower()[:5]:
                self._save_result(key, dv_funny_utterance, 'No.', None, None, val['vcap'], val['stt']['text']) # select 
                continue

            # Get explanations
            dv_explanation = dv_respondent.explain_why_funny(dv_funny_utterance)
            d_explanation = d_respondent.explain_why_funny(d_funny_utterance)

            self._save_result(key, dv_funny_utterance, d_funny_utterance, dv_explanation, d_explanation, val['vcap'], val['stt']['text'])


    def _save_result(self, key, dv_funny_utterance, d_funny_utterance, dv_explanation, d_explanation, vcap, stt):
        self.df.loc[len(self.df.index)] = [
            key,
            dv_funny_utterance, d_funny_utterance,
            dv_explanation, d_explanation,
            vcap, stt
        ]
        self.df.to_json(self.prompt_result_path, orient='index', indent=2)


class Respondent:

    def __init__(self, lm):
        self.lm = lm

    def find_funny_utterance(self):
        prompt = self._funny_utterance_prompt()
        return self.lm(prompt, engine="text-davinci-003", temperature=0)

    def _funny_utterance_prompt(self):
        raise NotImplementedError()

    def explain_why_funny(self, funny_utterance):
        prompt = self._explain_why_funny_prompt(funny_utterance)
        return self.lm(prompt, engine="text-davinci-003", temperature=0.3)

    def _explain_why_funny_prompt(self, funny_utterance):
        raise NotImplementedError()


class VisualDialogueRespondent(Respondent):
    """Respondent with visual description"""

    def __init__(self, lm, visual_description: str, utters):
        super().__init__(lm)
        self.visual_description = f" Description : '{visual_description}' "
        dialogue = f" Transcript :"
        for utter in utters:
            dialogue += f" '{utter}',"
        self.dialogue = dialogue[:-1] + '.'

    def _funny_utterance_prompt(self):
        return f"In this task, you will see a description and a transcript of a video. Find a funny utterance " \
               f"in the given transcript. If there is no funny utterance, answer 'No.'"\
               f"{self.visual_description}{self.dialogue} Funny utterance: "

    def _explain_why_funny_prompt(self, funny_utterance):
        return f"In this task, you will see a description and " \
               f"a transcript of a funny video. Explain why the video is funny." \
               f"{self.visual_description}{self.dialogue}. Explanation in one sentence: "  


class DialogueRespondent(Respondent):
    """Respondent with only transcript"""

    def __init__(self, lm, utters):
        super().__init__(lm)
        dialogue = f" Transcript :"
        for utter in utters:
            dialogue += f" '{utter}',"
        self.dialogue = dialogue[:-1] + '.'

    def _funny_utterance_prompt(self):
        return f"In this task, you will see a transcript of a video. Find a funny utterance in the given transcript. " \
               f"If there is no funny utterance, answer 'No.' "\
               f"{self.dialogue} Funny utterance: "
        
    def _explain_why_funny_prompt(self, funny_utterance):
        return "In this task, you will see a transcript of a funny video. " \
               f"Explain why the video is funny. {self.dialogue}. Explanation in one sentence: "


class LanguageModel:
    def __init__(self, key):
        openai.api_key = key

    def __call__(self, prompt, engine="text-davinci-003", temperature=0.1) -> str:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            top_p=1,
            max_tokens=200
        )

        return response['choices'][0]['text'].replace('\n', '').strip('\'\" ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_key', type=str, help='OPENAI key to use api')
    parser.add_argument('--video_ids', type=str, help='Directory of torch .pt file consisting of video ids', default='./video_ids.pt')
    parser.add_argument('--video_info', type=str, help='Directory of results of speech-to-text and video captioning', default='./videos/info.json')
    parser.add_argument('--prompt_result_path', type=str, help='Directory to store the result of pipeline', default='pipeline_result.json')
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    lm = LanguageModel(api_key)
    usecase = FunnyUtteranceFiltering(
        lm=lm,
        args=args
    )
    usecase()
        
