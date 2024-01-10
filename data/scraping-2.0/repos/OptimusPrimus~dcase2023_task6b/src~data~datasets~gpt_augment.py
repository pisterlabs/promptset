import os

import torch
import json
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from sacred import Ingredient
from utils.directories import directories, get_dataset_dir

gpt_augment = Ingredient('gpt_augment', ingredients=[directories])


@gpt_augment.config
def config():
    folder_name = 'clotho_gpt'
    api_key = ''

class GPTAugment(torch.utils.data.Dataset):

    def __init__(self, dataset, p=0.0, num_variations=5, add_keywords=True):
        self.dataset = dataset
        self.p = p
        self.num_variations = num_variations
        self.variations = get_variations(dataset, num_variations=5, add_keywords=add_keywords, create=False)

        self.variations = {s['idx']: s for s in self.variations }


    def __getitem__(self, item):

        if self.p == 0 or torch.rand((1,)).item() > self.p:
            return self.dataset[item]
        else:
            j = torch.randint(0, self.num_variations, (1,)).item()
            s = self.dataset[item].copy()
            s['caption'] = self.variations[s['idx']]['variations'][j]
            return s

    def __len__(self):
        return len(self.dataset)


def infinite_request(query, max_attempts=10):
    i = 0
    import time
    while i < max_attempts:
        try:
            return openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": query}
                ],
                timeout=10
            )
        except:
            print(f'Failed waiting {5 + 2**i}s')
            time.sleep(5 + 2**i)
            print('Retry')
        i += 1

    raise

@gpt_augment.capture
def get_variations(ds, num_variations=5, add_keywords=False, create=False, folder_name='clotho_gpt'):

    # for caching
    cached_path = os.path.join(get_dataset_dir(), folder_name)
    if not os.path.exists(cached_path):
        os.mkdir(cached_path)

    if os.path.exists(os.path.join(cached_path, 'variations.json')):
        with open(os.path.join(cached_path, 'variations.json'), 'r')as f:
            return json.load(f)
    else:
        print(f'variations.json not found, create in {cached_path}')

    # construct the prompt
    if add_keywords:
        instruction = f'I will give a description of an audio recording and tags associated with the audio recording. ' \
                      f'Generate {num_variations} audio caption{"s" if num_variations > 1 else ""} describing the sound events. ' \
                      'Each audio caption should be one sentence with less than 20 words. ' \
                      'Use grammatical subject-verb-object sentences. ' \
                      'Do not include names of cities, countries, and persons. ' \
                      'Do not include the time. ' \
                      'Do not include recording device names. ' \
                      'Do not write introductions or explanations. ' \
                      'Do not use “heard”, “recorded”. ' \
                      'Start each output sentence with its index. '
    else:
        instruction = f'Rephrase the following audio description {num_variations} time{"s" if num_variations > 1 else ""}. ' \
                      'Vary the sentence length, but do not use more than 20 words. ' \
                      'Try not to use the same words.'

    variations = []
    from tqdm import tqdm

    for _, s in tqdm(enumerate(ds)):
        # create the query
        id = str(s['idx']) + '_' + "_".join(s['path'].split(os.path.sep)[-2:] + [str(add_keywords), str(num_variations)]) +'.json'

        content = '\'' + s["caption"] + '\'' + (' [' + s["keywords"].replace(';', ', ') + ']' if add_keywords else '')
        query = instruction + content

        p = os.path.join(cached_path, id)
        if os.path.exists(p):
            print(f'Loading {p}.')
            with open(p, 'r') as f:
                response = json.load(f)
        else:
            if not create:
                continue
            response = infinite_request(query)
            with open(p, 'w') as f:
                json.dump(response, f)



        result = {
            'idx': s['idx'],
            'path': s['path'],
            'caption': s['caption'],
            'query': query,
            'response': response,
            'variations': response['choices'][0]["message"]["content"] if len(response['choices']) > 0 else None,
            'finish_reason': response['choices'][0]["finish_reason"] if len(response['choices']) > 0 else None,
            'usage': response['usage']["total_tokens"]
        }

        variations.append(result)

    for s in variations:
        captions = s['variations'].replace('\n\n','\n').split('\n')
        if len(captions) != num_variations:
            pass
            # print('invalid number of variations')

        if all([c[0] == str(i+1) for i, c in enumerate(captions)]):
            captions = [c[1:] for c in captions]
        elif all([c[0] in ['a', 'b', 'c', 'd', 'e'] for i, c in enumerate(captions)]):
            captions = [c[1:] for c in captions]
        elif any([c.startswith('one: ') for i, c in enumerate(captions)]):
            captions = [c.replace('one: ', '').replace('two: ', '').replace('three: ', '').replace('four: ', '').replace('five: ', '') for c in captions]
        else:
            pass
            # print('No enumeration')

        if all([c[0:2] in ['. ', ') '] for i, c in enumerate(captions)]):
            captions = [c[2:] for c in captions]
        elif all([c[0] in ['.', ')'] for i, c in enumerate(captions)]):
            captions = [c[1:] for c in captions]
        else:
            pass
            # print('Malformed enumeration.')

        s['variations'] = captions

    with open(os.path.join(cached_path, folder_name, 'variations.json'), 'w')as f:
        json.dump(variations, f)

    return variations


if __name__ == "__main__":
    import openai

    openai.api_key = ''  # insert your key here

    from sacred import Experiment
    from data.datasets.clotho_v2 import clotho_v2, get_clotho_v2

    ex = Experiment('test', ingredients=[clotho_v2, gpt_augment])

    @ex.main
    def main_():

        mode = "train"
        add_keywords = True
        num_variations = 5
        clotho = get_clotho_v2(mode)

        variations = get_variations(clotho, num_variations=num_variations, add_keywords=add_keywords, create=False)

        print("Finished creating wavecaps_gpt")
        print("Created ", len(variations), "augmentations")


    ex.run()
