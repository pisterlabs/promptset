import json
import openai
import os
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import torch
import signal
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from glob import glob


class Diarization:
    def __init__(self, lm, meta_path):
        self.lm = lm
        self.metas = [i for i in sorted(glob(str(Path(meta_path))), key=lambda x:x.lower()) if os.path.exists(i)]
        self.vids = [i.split('/')[-2] for i in self.metas]
        self.js = {}
        
    def _get_transcript(self, meta):
        transcript = ''
        transcript_list = []
        for i, scene in enumerate(meta):
            for k, utter in enumerate(scene['text']):
                if len(utter) > 0:
                    transcript += f'{utter}' + '\n'
                    transcript_list.append(utter)
        transcript = 'Transcript: ' + transcript
        return transcript, transcript_list, len(transcript_list)
    

    def _make_num_speaker_prompt(self, transcript):
        prompt1 = f'Based on the context of the given transcript, how many speakers are there? Please provide only the number.\n'\
                f'{transcript}'\
                f'What is the most likely number of speakers in the given transcript? Please provide only the most likely number and no other explanation.'
        prompt1 = [
                    {"role" : "system", "content" : "You are a helpful assistant."},
                    {"role" : "user", "content" : prompt1}
                ]
        res1 = self._run_lm(prompt1)
        return prompt1, res1
    

    def _make_diarization_prompt(self, prompt1, res1):
        prompt2 = f'Based on the context of the given transcript and your estimated number of speakers, please assign speakers to the transcript.'
        prompt2 = [
                    {"role" : "system", "content" : "You are a helpful assistant."},
                    {"role" : "user", "content" : prompt1[-1]['content']},
                    {"role" : "user", "content" : res1},
                    {"role" : "user", "content" : prompt2}
                ]
        res2 = self._run_lm(prompt2)
        return prompt2, res2
    

    def _make_diarization_prompt_sequentially(self, transcript_list, prompt1, res1):
        output = ['Speaker 1'] 
        if len(transcript_list) > 1:
            prompt2 = [
                    {"role" : "system", "content" : "You are a helpful assistant."},
                    {"role" : "user", "content" : prompt1},
                    {"role" : "system", "content" : res1},
                    {"role" : "user", "content" : f"Please let me know the most likely speaker number given a sentence in the transcript. "\
                                                    f"Example) Sentence: {transcript_list[0]} Speaker Number : Speaker 1. Sentence: {transcript_list[1]}"},
                    ]
            res2 = self._run_lm(prompt2)
            output.append(res2)
            prompt2.append({"role": "system", "content": res2})
        for s in transcript_list[2:]:
            prompt2.append({"role" : "user", "content" : f'Sentence: {s}'})
            res2 = self._run_lm(prompt2)
            output.append(res2)
            prompt2.append({"role": "system", "content": res2})
        return prompt2, output 


    def _run_lm(self, prompt):
        return self.lm(prompt)


    def _is_not_dialogue(self, res):
        if 'one' in res or '1' in res or '0' in res:
            return True
        return False

    def find_number(self, string):
        for word in string.split():
            if word.isnumeric():
                return int(word)


    def get_speaker(self, string):
        pattern = r'speaker\s*\d+'
        indices = []
        for match in re.finditer(pattern, string.lower()):
            indices.append((match.start(), match.end()-1, match.group()))
        utters = ['','','','','','','','','','']
        for i, (s, e, speaker) in enumerate(indices):
            num = self.find_number(speaker) - 1
            if i == len(indices) - 1 :
                n_s = len(string) -1
            else:
                n_s = indices[i+1][0]
            utters[num] += ' ' + string[s:n_s]
        while len(utters) > 0 and len(utters[-1]) == 0:
            utters.pop()
        return utters

    def get_speaker_seq(self, string):
        pattern = r'speaker\s*\d+'
        try:
            match = list(re.finditer(pattern, string.lower()))[0]
            for word in match.group().split():
                if word.isnumeric():
                    return int(word)
        except:
            return 0

    
    def _post_processing(self, meta, num, gpt_result=None, sequential=False):
        # monologue
        if num == 1:
            for i, scene in enumerate(meta):
                diar = []
                for utter in scene['text']:
                    if len(utter) == 0:
                        diar.append("")
                    else:
                        diar.append("1")
                meta[i]['diar'] = diar
            return meta
        
        # dialogue
        elif not sequential:
            # split result and assign speakers
            speakers = self.get_speaker(gpt_result)
            
            # check whether the result is okay
            total = 0
            for s in speakers:
                total += len(s)
            if total < num:
                return None
            
            # if ok, update meta data
            for i, scene in enumerate(meta):
                diar = []
                for utter in scene['text']:
                    if len(utter) == 0:
                        diar.append('')
                        continue
                    for speaker, string in enumerate(speakers):
                        if utter.lower().strip() in string.lower():
                            diar.append(f'{speaker + 1}')
                            break
                meta[i]['diar'] = diar
            return meta
        # dialogue with sequential result
        else:
            res = []
            b_speaker = 1
            for g_res in gpt_result:
                speaker = self.get_speaker_seq(g_res)
                if speaker > 0:
                    res.append(str(speaker))
                    b_speaker = speaker
                else:
                    res.append(str(b_speaker))

            mark = 0
            for i, scene in enumerate(meta):
                diar = []
                for utter in scene['text']:
                    if len(utter) > 0:
                        diar.append(str(res[mark]))
                        mark += 1
                    else:
                        diar.append('')
                meta[i]['diar'] = diar
            return meta
        
    
    def __call__(self):
        
        for i, aud in tqdm(enumerate(self.metas), total= len(self.metas)):
            
            with open(aud) as f:
                meta = json.load(f)
            
            transcript, transcript_list, num = self._get_transcript(meta)
            
            # if no transcript
            if num == 0: 
                continue
            
            # if only one utterance
            if num == 1:
                meta = self._post_processing(meta, 1)
                
            # if there are more than two utterances
            else:
                prompt1, res1 = self._make_num_speaker_prompt(transcript)
                
                # if speakers < 2
                if self._is_not_dialogue(res1.lower()):
                    meta = self._post_processing(meta, 1)
                    
                # speakers >= 2
                else:
                    prompt2, res2 = self._make_diarization_prompt(prompt1, res1)
                    meta = self._post_processing(meta, num, res2)
                    if not meta:
                        prompt3, res3 = self._make_diarization_prompt_sequentially(transcript_list, prompt1, res1)
                        meta = self._post_processing(meta, num, res3, sequential=True)
            
            with open(aud, 'w') as f:
                json.dump(meta, f, indent=2)
    
    
class LanguageModel:
    def __init__(self, key):
        openai.api_key = key
        signal.signal(signal.SIGALRM, self.alarm_handler)

    def __call__(self, prompt, engine="text-davinci-003", temperature=0.1) -> str:  
        while True:
            try:
                res = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo",
                    messages=prompt,
                    temperature=0
                )
                return res['choices'][0]['message']['content']
            except Exception as e:
                print(e)
                continue
    
    def alarm_handler(signum, frame):
        raise Exception('timeout')


def run_diarization(root_dir, segmentation_filename):  
    api_key = os.getenv("OPENAI_API_KEY")
    gpt3_api = LanguageModel(api_key)
    meta_path = root_dir + '/*/' + segmentation_filename
    usecase = Diarization(
        lm=gpt3_api,
        meta_path=meta_path,
    )
    usecase()
        
