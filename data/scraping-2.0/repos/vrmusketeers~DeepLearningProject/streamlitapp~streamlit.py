#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 23:44:19 2021

@author: Shiv Kumar Ganesh
"""

# Installing Dependencies
import os

# Required for the image rendering
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['WANDB_API_KEY']='d74036002407b5a3ba2ad5be469b3793afde3eb2'

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024, DiscreteVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer
from torchvision.utils import make_grid
import torch
import wandb


# Downloading the artifact from wandb
@st.cache
def downloading_model():
    run = wandb.init()
    artifact = run.use_artifact('sjsu-cmpe-258-musketeers/cubicasa-dalle/trained-dalle:v5', type='model')
    artifact_dir = artifact.download()

# Load the downloaded dalle model for the floor plan generator
@st.cache
def loading_model():
    loaded_obj = torch.load(str('./artifacts/trained-dalle:v5/dalle-final.pt'), map_location='cpu')
    dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']
    vae = DiscreteVAE(**vae_params)
    dalle = DALLE(vae=vae, **dalle_params)
    dalle.load_state_dict(weights)
    return dalle

# Generating the image and displaying it on the UI
def generate_image_and_display():
    st.write('Dalle designed floorplan!!')
    image = Image.open('mygraph.png')
    st.image(image, caption='Your Dream home is ready ;)')
    os.remove('mygraph.png')

# Dalle generating the floor plan
def generate_floow_plan(input_text, dalle):
    with st.spinner('Waiting for Model to Run...'):
        descriptions = list(filter(lambda t: len(t) > 0, input_text))
        text_token = tokenizer.tokenize(descriptions, 256).squeeze(0)
        image = dalle.generate_images(text_token[:1], filter_thres = 0.9)
        st.success('Done!')

    with st.spinner('Waiting for Visualization to Run...'):
        arr_ = np.squeeze(image[0].permute(1,2,0).cpu())
        print(input_text)
        plt.imshow(arr_)
        plt.axis('off')
        plt.savefig("mygraph.png")

downloading_model()
dalle_t = loading_model()

st.title('Enter the description of the Floor plan you want')
st.write('For Example: One bedroom with two kitchen....')
text_input = st.text_input('Enter your description', value='', type='default')

if st.button('Get Floor plan'):
    generate_floow_plan(text_input, dalle_t)
    generate_image_and_display()
