import json
import openai
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import os
import torch
from sacrebleu import BLEU
from torchmetrics import BLEUScore
from collections import OrderedDict
import signal
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from glob import glob
from tqdm import tqdm
import librosa
from openlimit import ChatRateLimiter
import multiprocessing as mp
import time
#args -> root_dir, out_pth

class Explanationer:
    
    def __init__(self, args):
        
        self.output_pth = Path(args.out_pth)
        self.root_dir = Path(args.root_dir)
        
        self.explanations = {}
        self.viddirs = sorted([i for i in self.root_dir.glob('*') if i.is_dir()], key=lambda x: str(x).lower())
        self.vids = [i.name for i in self.viddirs]
        
        

    def _run_lm(self, prompt):
        lm = LanguageModel(os.getenv("OPENAI_API_KEY"))
        return lm(prompt)
    
    
    # make scenes
    def _get_meta(self, vid):
        
        # meta
        meta_pth = f'{self.root_dir}/{vid}/segments.json'
        with open(meta_pth) as f:
            meta_dict = json.load(f)
        
        audtag_pth = f'{self.root_dir}/{vid}/audtag.pt'
        audtag = torch.load(audtag_pth)
        audtag = [i[0] for i in audtag[:3] if i[1] >= 0.1]
        
        # gather scene descriptions
        scenes = []
        n = 1
        
        for meta in meta_dict:
            for i in range(len(meta['start'])):
                s, e = meta['start'][i], meta['end'][i]
                if e - s < 0.01:
                    meta['vcap'].insert(i, '')
                v = meta['vcap'][i]
                d = meta['diar'][i]
                t = meta['text'][i]
                if len(v) > 0:
                    if len(t) > 0 :
                        p = f"Scene:\nSpeaker {d}: \"{t}\"\n{v[0].upper()}{v[1:]}.\n\n"
                    else:
                        p = f"Scene:\n{v[0].upper()}{v[1:]}.\n\n"
                n += 1
                scenes.append([(s, e), (v, d, t), p])
        
        # concatenate scene descriptions
        scene_desc = ''
        for _, _, p in scenes:
            scene_desc += p
        
        # concatenate audio descriptions
        audtag = ','.join(audtag)
        
        return scene_desc, audtag
    

    def call_gpt3_api(self, lm, prompt, prompt_list=None, temp=0):
        result = lm(prompt, temp)
        return result
    
    
    def _explanation_prompt_w_audio(self, lm, scenes, audio_cap):
        prompt = "Please generate a sentence of explanation of why a video is funny, given "\
                "visual descriptions and dialogue(or monologue) from the video. "\
                "Explain as if you were watching the video. "\
                "Visual descriptions and utterances will be given in chronological order.\n\n"\
                f"Visual description and utterances (chronologically): \n\n({audio_cap})\n\n{scenes}Explanation: "
        result = lm(prompt)
        return prompt, result
    
    
    def _run_video(self, vid):
        time.sleep(0.5)
        scene_desc, audtag = self._get_meta(vid)
        lm = LanguageModel(os.getenv("OPENAI_API_KEY"))
        prompt, explanation = self._explanation_prompt_w_audio(lm, scene_desc, audtag)
        return (vid, {'prompt': prompt, 'res': explanation})
    
    
    def __call__(self):
        #for i, meta_pth in tqdm(enumerate(self.metas)):
        len_chunk = 3
        idx_list = range(0, len(self.vids), len_chunk)
        for i, start in enumerate(idx_list):
            print(f'Progress : {i} / {len(idx_list)}')
            p = mp.Pool(10)
            sub_vids = self.vids[start : start+len_chunk]
            result = tqdm(p.imap(self._run_video, sub_vids), total=len(sub_vids))
            p.close()
            p.join
            self.explanations.update(dict(result))
            print(f'Store {i}th iteration result' )
            with open(self.output_pth, 'w') as f:
                json.dump(self.explanations, f, indent=2)
        

class LanguageModel:
    def __init__(self, key):
        openai.api_key = key
        signal.signal(signal.SIGALRM, self.alarm_handler)

    def __call__(self, prompt, temperature=0.3) -> str:  
        while True:
            try:
                res = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=300,        
                    )
                return res['choices'][0]['text']
            except Exception as e:
                print(e)
                continue
            

    def alarm_handler(signum, frame):
        raise Exception('timeout')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_dir', default='/gallery_louvre/dayoon.ko/dataset/videohumor')
    parser.add_argument('--meta', default='/gallery_louvre/dayoon.ko/research/mmvh/videohumor/exp_output/audio_caption/audio_caption.json')
    parser.add_argument('--out_pth', default='./explanations.json')
    args = parser.parse_args()

    # alarm
    signal.signal(signal.SIGALRM, alarm_handler)
    
    # output path
    usecase = Explanation(
        args=args
    )
    usecase()