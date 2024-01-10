#!/usr/bin/env python
# coding: utf-8

import os, glob
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import re
import json
import time
import random
from openai import OpenAI

# actual distribution of Yelp reviews
target_pcts_per_sentiment = {
    'Very positive':.454797,
    'Positive':.248251,
    'Neutral':.115715,
    'Negative':.098499,
    'Very negative':.082738
}

# based on target distribution mimicking Yelp reviews
SENTIMENTS = ["Very positive"]*55 + ["Positive"]*30 + ["Neutral"]*14 + ["Negative"]*12 + ["Very negative"]*10

STARS = ["$ ($10 and under)",
"$$ ($10-$25)",
"$$$ ($25-$45)",
"$$$$ ($50 and up)"]

CUISINES = ['Chinese',
 'Mexican',
 'Thai',
 'Vietnamese',
 'Indian',
 'Korean',
 'Latin American',
 'Mediterranean',
 'Irish',
 'Japanese',
 'Greek',
 'Soul food',
 'French',
 'Cajun',
 'Creole',
 'Italian',
 'Southern',
 'Spanish',
 'Cuban',
 'American']

FOCUSES = ['staff','waitstaff','employees','waiter','waitress',
           'food','drinks','main courses','appetizers','desserts',
           'place', 'spot', 'atmosphere', 'experience', 'ambiance']

PROMPT_TEMPLATES = [
    'A customer posted the following restaurant review to an online restaurant review website: <span class="headline" title="%s review about a %s %s restaurant, focused on the %s">',
    'Write a %s review of a %s %s restaurant, focusing on the %s',
    'Give an example of a %s review of a %s %s restaurant'
]

NEUT_PROMPT_TEMPLATES = [
    'A customer posted the following restaurant review to an online restaurant review website: <span class="headline" title="Review about a %s %s restaurant, focused on the %s">',
    'Write a review of a %s %s restaurant, focusing on the %s',
    'Give an example of a review of a %s %s restaurant'
]

def main(api_key, models, out_dir, debug):
    
    # create client
    client = OpenAI( 
          api_key = api_key
    )
    
    if debug:
        prompt_templates = PROMPT_TEMPLATES[:1]
        neut_prompt_templates = NEUT_PROMPT_TEMPLATES[:1]
        sentiments = SENTIMENTS[:1]
        stars = STARS[:1]
        cuisines = CUISINES[:1]
        focuses = FOCUSES[:1]
    else:
        prompt_templates = PROMPT_TEMPLATES
        neut_prompt_templates = NEUT_PROMPT_TEMPLATES
        sentiments = SENTIMENTS
        stars = STARS
        cuisines = CUISINES
        focuses = FOCUSES
    
    cnt = 0
    responses = []
    for model in models:
        for i, prompt_template in enumerate(prompt_templates):
            for sentiment in sentiments:
                for star in stars:
                    for cuisine in cuisines:
                        if i < 2:
                            focus = np.random.choice(a=focuses, size=1)[0]
#                             for focus in focuses:
                            if sentiment =='Neutral':
                                prompt = neut_prompt_templates[i] %(star,cuisine,focus)
                            else:
                                prompt = prompt_template %(sentiment,star,cuisine,focus)
                        else:
                            focus = 'none'
                            
                            if sentiment =='Neutral':
                                prompt = neut_prompt_templates[i] %(star,cuisine)
                            else:
                                prompt = prompt_template %(sentiment,star,cuisine)

                        entry = {}
                        entry['model'] = model
                        entry['prompt_ix'] = i
                        entry['cuisine'] = cuisine
                        entry['star'] = star
                        entry['sentiment'] = sentiment
                        entry['focus'] = focus
                        
                        try:
                            response = client.chat.completions.create(
                                  model=model,
                                  messages=[
                                      {"role": "user", "content": prompt}
                                  ],
                                  temperature=1,
                                  max_tokens=256,
                                  top_p=1,
                                  frequency_penalty=0,
                                  presence_penalty=0
                            )
                            text = response.choices[0].message.content
                        except:
                            text = 'error'
                        entry['review'] = text

#                         try:
#                             response = completions_with_backoff(
#                               model=model, max_tokens = 200,
#                               messages=[
#                                 {"role": "user", "content": prompt}
#                             ])
#                             text = response.choices[0].message.content#response["choices"][0]["message"]["content"]

#                         except:
#                             text = "error"
                        
                        responses.append(entry)
                        with open(os.path.join(out_dir, f'review_{cnt}.json'), 'w') as f:
                            f.write(json.dumps(entry))
                        cnt += 1
                        time.sleep(0.5)

#                         if cnt%200==0:
#                             file2write=open("log.txt",'w')
#                             file2write.write(str(cnt))
#                             file2write.close()
    df = pd.DataFrame(responses)
    print(cnt, len(responses))
    print(df.shape)
    print(df.head())
    print(df.tail())
    for col in df.columns:
        print(df[col].value_counts())        
    
    df.to_csv(os.path.join(out_dir, 'responses.csv'))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default='',
                        help='<ADD YOUR OWN API KEY>')
    parser.add_argument('--models', type=str, default='gpt-3.5-turbo-0613,gpt-3.5-turbo-1106',
                        help='models to prompt, separated by commas')
    parser.add_argument('--out_dir', type=str, default='../data/gpt_output',
                        help='directory to save output to')
    parser.add_argument('--debug', action='store_true',
                        help='whether to run on subset of data for debugging purposes')
    args = parser.parse_args()
    if not args.debug:
        print("\n******WARNING****** DEBUG MODE OFF!")
    else:
        print("\nRunning in debug mode; will prompt only 1 model and varying only 1 element per field.")
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    models = args.models.split(',')
    if args.debug:
        models = models[:1]
    print(f'\nWill prompt the following {len(models)} models:', models)
        
    main(args.api_key, models, args.out_dir, args.debug)
    