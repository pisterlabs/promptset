import os
import sys

import re
import numpy as np
import time

import random
from tqdm import tqdm
import shutil

import clip
import openai
sys.path.append('pixray')
import pixray

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

min_time_between_calls = 2.

time_previous_call = time.perf_counter_ns()


create_images = False

def calc_children_concepts(concept, prompt="In a comma-separated list, give me ten one-word objects related to '{}':"):
    global time_previous_call
    time_call = time.perf_counter_ns()
    
    dt = (time_call-time_previous_call)/1000000000
    if dt < min_time_between_calls:
        time.sleep(min_time_between_calls-dt)
    # print((time.perf_counter_ns()-time_previous_call)/1000000000)
    
    prompt = prompt.format(concept)
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=prompt,
        temperature=0,
        max_tokens=50,
    )
    time_previous_call = time.perf_counter_ns()
    
    response = response["choices"][0].text
    response = response.strip().replace(' ', '')
    
    response = response.replace(',', '\n')
    response = re.sub(r"\d+\.", "", response)
    response = response.lower()
    
    return np.array(response.replace(' ', '').split('\n'))


# calc_children_concepts('dog')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def calc_text_features(texts, normalize=True):
    x = clip.tokenize(texts).to(device)
    with torch.no_grad():
        x = model.encode_text(x).detach()
    if normalize:
        x = x/x.norm(dim=-1, keepdim=True)
    return x.detach().cpu()

def make_image_with_parents(noun, potential_parents=None):
    kwargs = dict(prompts="an icon of a {} #pixelart".format(noun), 
                  drawer="fast_pixel",
                  quality="better",
                  size=[128, 128],
                  pixel_size= [16, 16],
                  scale=1, 
                  iterations=100, 
                  init_noise='gradient',
                  init_image=None, 
                  outdir=f'outputs/{noun}/ACTUAL', 
                  make_video=False, 
                  save_intermediates=True)
    
    # print(f'creating images of {noun} with potential parents: {potential_parents}')
    
    if potential_parents is None:
        pixray.run(**kwargs)
    else:
        losses = []
        for parent in potential_parents:
            kwargs['init_image'] = f'outputs/{parent}/ACTUAL/output.png'
            kwargs['outdir'] = f'outputs/{noun}/{parent}/'
            pixray.run(**kwargs)
            losses.append(pixray.best_loss.item())
        best_parent = potential_parents[np.argmin(np.array(losses))]
        shutil.copytree(f'outputs/{noun}/{best_parent}', f'outputs/{noun}/ACTUAL')

random.seed(1)
def open_ended_run(seed_concept='dog', novelty_metric=None, n_children=2, n_gen=100):
    """
    novelty_metric can be either min or avg
    """
    pop = [seed_concept]
    pop_features = calc_text_features([seed_concept])
    
    pop_set = {seed_concept}
    
    parents = set()
    
    min_dists = []
    avg_dists = []
    
    concept = seed_concept
    
    if create_images:
        make_image_with_parents(seed_concept, potential_parents=None)
    
    
    children_data = {}
    parent_data = {}
    
    for i in tqdm(range(n_gen)):
        parents.add(concept)
        
        children = calc_children_concepts(concept)
        children = np.array([c for c in children if c not in pop_set])
        
        if len(children)<n_children:
            continue
        
        children_features = calc_text_features(children)
        
        dots = children_features@pop_features.T
        min_dist_to_pop = dots.max(dim=-1).values
        avg_dist_to_pop = dots.mean(dim=-1)
        
        if novelty_metric is None:
            idx = np.arange(10)
        elif novelty_metric == 'min':
            idx = min_dist_to_pop.argsort(dim=0)
        elif novelty_metric == 'avg':
            idx = avg_dist_to_pop.argsort(dim=0)
        idx = idx[:n_children]
        
        
        min_dists.append(min_dist_to_pop[idx].mean().item())
        avg_dists.append(avg_dist_to_pop[idx].mean().item())
        
        potential_img_seeds = np.array(pop)[np.random.permutation(len(pop))[:2]]
        
        pop.extend(children[idx])
        pop_set.update(children[idx])
        pop_features = torch.cat([pop_features, children_features[idx]], dim=0)
        
        print(f'Chosen parent: {concept} || Children: {children[idx]}')
        
        for child in children[idx]:
            if concept not in children_data:
                children_data[concept] = []
            children_data[concept].append(child)
            parent_data[child] = concept
            
        if create_images:
            make_image_with_parents(child, potential_parents=potential_img_seeds)
            
        concept = random.choice(list(pop_set.difference(parents)))
            
        torch.save(pop, 'outputs/pop.pth')
        torch.save(children_data, 'outputs/children_data.pth')
        torch.save(parent_data, 'outputs/parent_data.pth')
        
        # rather than randomly sampling a concept from the set, we should use a queue which priorirtizes novel concepts
        # rather than computing how novel a concept is each time (which takes n^2).
        # we only use how novel it is WHEN it was originally formulated as a child. This is already being done and has no overhead.
    
        # print((pop_features@pop_features.T).mean())
        # print(min_dist_to_pop[idx].mean().item())
        # print(avg_dist_to_pop[idx].mean().item())
        
        # if i%30==0:
        #     print(f'Chosen parent: {concept} || Children: {children[idx]}')
        #     plt.plot(min_dists, label='min')
        #     plt.plot(avg_dists, label='avg')
        #     plt.show()
        
        
        
if __name__=="__main__":
    open_ended_run('tree', novelty_metric='min', n_gen=100000)