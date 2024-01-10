# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import os
import pickle
import torch
import openai
from sentence_transformers import SentenceTransformer

label_encoder_classes = {0: 'Account Management', 1: 'Ads', 2: 'Badges/Emotes', 
                         3: 'Bits', 4: 'Channel Page', 5: 'Channel Points', 
                         6: 'Charity', 7: 'Chat', 8: 'Creator Camp', 
                         9: 'Creator Dashboard', 10: 'Creator Dashboard: Stream Manager', 
                         11: 'Creators and Stream Features', 12: 'Customer Experience', 
                         13: 'Developers', 14: 'Discover', 15: 'Extensions', 16: 'IGDB', 
                         17: 'IRL Events and Merch', 18: 'Localization', 19: 'Moderation', 
                         20: 'Purchase Management', 21: 'Safety', 22: 'Subscriptions', 
                         23: 'Twitch Applications: Consoles', 24: 'Twitch Applications: Mobile', 
                         25: 'Twitch Applications: TV Apps', 26: 'Twitch Studio', 
                         27: 'User Accessibility', 28: 'Video Features', 29: 'Video Performance'}

from torch import nn

class NN_CLF_GPT(nn.Module):
    def __init__(self, input_size=1536, output_size=30):
        super(NN_CLF_GPT, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
class NN_CLF_BERT(nn.Module):
    def __init__(self, input_size=384, output_size=30):
        super(NN_CLF_BERT, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
class NN_REG_GPT(nn.Module):
    def __init__(self, input_size=1536):
        super(NN_REG_GPT, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)  # Output size is 1 for regression
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
class NN_REG_BERT(nn.Module):
    def __init__(self, input_size=384):
        super(NN_REG_BERT, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)  # Output size is 1 for regression
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
 
class NN_REGSIF_GPT(nn.Module):
    def __init__(self, input_size=1536, num_classes=30):
        super(NN_REGSIF_GPT, self).__init__()
        # Shared layers
        self.base_layer1 = nn.Linear(input_size, 128)
        self.base_layer2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()

        # Regression head
        self.regression_head = nn.Linear(64, 1)  # Output one value for regression

        # Classification head
        self.classification_head = nn.Linear(64, num_classes)  # Output for each class

    def forward(self, x):
        # Shared layers
        x = self.relu(self.base_layer1(x))
        x = self.relu(self.base_layer2(x))

        # Regression and classification heads
        regression_output = self.regression_head(x)
        classification_output = self.classification_head(x)

        return regression_output, classification_output
       
class NN_REGSIF_BERT(nn.Module):
    def __init__(self, input_size=384, num_classes=30):
        super(NN_REGSIF_BERT, self).__init__()
        # Shared layers
        self.base_layer1 = nn.Linear(input_size, 128)
        self.base_layer2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()

        # Regression head
        self.regression_head = nn.Linear(64, 1)  # Output one value for regression

        # Classification head
        self.classification_head = nn.Linear(64, num_classes)  # Output for each class

    def forward(self, x):
        # Shared layers
        x = self.relu(self.base_layer1(x))
        x = self.relu(self.base_layer2(x))

        # Regression and classification heads
        regression_output = self.regression_head(x)
        classification_output = self.classification_head(x)

        return regression_output, classification_output
    
nn_clf_gpt = NN_CLF_GPT()
nn_clf_bert = NN_CLF_BERT()
nn_reg_gpt = NN_REG_GPT()
nn_reg_bert = NN_REG_BERT()
nn_regsif_gpt = NN_REGSIF_GPT()
nn_regsif_bert = NN_REGSIF_BERT()

models = {}
ml_model_names, dl_model_names = [], []
for model_name in os.listdir():
    model_first_name = model_name.split('.')[0]
    if '.pkl' == model_name[-4:] or '.pickle' in model_name[-7:]:
        ml_model_names.append(model_first_name)
        with open(f'{model_name}', 'rb') as f:
            models[model_first_name] = pickle.load(f)
    elif '.pt' == model_name[-3:] or '.pth' in model_name[-4:]:
        dl_model_names.append(model_first_name)
        if model_first_name == 'neural_network_classification_GPT':
            nn_clf_gpt.load_state_dict(torch.load(f'{model_name}'))
            models[model_first_name] = nn_clf_gpt.to('cpu').eval()
        if model_first_name == 'neural_network_classification_BERT':
            nn_clf_bert.load_state_dict(torch.load(f'{model_name}'))
            models[model_first_name] = nn_clf_bert.to('cpu').eval()
        if model_first_name == 'neural_network_regression_GPT':
            nn_reg_gpt.load_state_dict(torch.load(f'{model_name}'))
            models[model_first_name] = nn_reg_gpt.to('cpu').eval()
        if model_first_name == 'neural_network_regression_BERT':
            nn_reg_bert.load_state_dict(torch.load(f'{model_name}'))
            models[model_first_name] = nn_reg_bert.to('cpu').eval()
        if model_first_name == 'regsification_GPT':
            nn_regsif_gpt.load_state_dict(torch.load(f'{model_name}'))
            models[model_first_name] = nn_regsif_gpt.to('cpu').eval()
        if model_first_name == 'regsification_BERT':
            nn_regsif_bert.load_state_dict(torch.load(f'{model_name}'))
            models[model_first_name] = nn_regsif_bert.to('cpu').eval()

st.write('All models loaded.')

sbert_model_name = 'paraphrase-MiniLM-L6-v2'
device = 'cpu'
sbert = SentenceTransformer(sbert_model_name, device=device)

def embed_text_openai(text, model="text-embedding-ada-002"):
    client = openai.OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    text = str(text).replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def embed_text_BERT(text, emb_model=sbert):
    embeddings = emb_model.encode(text, convert_to_tensor=True).cpu().numpy()
    return embeddings

def inverse_transform_prediction(normalized_prediction, min_val= 0.0, max_val=18563.0):
    raw_prediction = normalized_prediction * (max_val - min_val) + min_val
    return raw_prediction

def get_prediction(text, model_name, emb_type):
    if emb_type == 'openai':
        emb = embed_text_openai(text)
    elif emb_type == 'bert':
        emb = embed_text_BERT(text)
    else:
        raise ValueError(f'Unknown embedding type: {emb_type}')
    
    if model_name in ml_model_names:
        model = models[model_name]
        pred = model.predict([emb])[0]
        return pred
    elif model_name in dl_model_names:
        if 'classification' in model_name:
            model = models[model_name]
            emb = torch.tensor(emb).float().to('cpu')
            outputs = model(emb)
            _, predicted = torch.max(outputs.data, 0)
            pred = predicted.item()
            return pred
        elif 'regression' in model_name:
            model = models[model_name]
            emb = torch.tensor(emb).float().to('cpu')
            pred = model(emb).detach().cpu().numpy()[0]
            return pred
    else:
        raise ValueError(f'Unknown model name: {model_name}')
        
def run_app():
    st.title('TwitchSight: Predictive Modeling and Analysis of Twitch User Ideas')
    st.header('Reza Khan Mohammadi, Patrick Govan, and Josiah Hill')
    
    st.write(
        'The project "TwitchSight" is currently focusing on predicting the popularity\
        of user ideas on Twitch and classifying them into respective themes. \
        We have scraped data from the Twitch UserVoice platform, resulting in 13,233\
        ideas spanning across 30 categories. We have taken this data and trained a\
        number of models to assign each idea into one of the UserVoice categories. \
        With this web app, you can classify a new idea into an appropriate category\
        using any of our models.'
        )
        
    text = st.text_input('UserVoice Idea', 'Enter a new prompt to classify.')
  
    model_name = st.selectbox(
        'Select a model to predict your data:',
        ml_model_names + dl_model_names)
    
    emb_type = "openai" if "GPT" in model_name else "bert"
    
    if st.button('Execute Model'):
        if 'classification' in model_name:
            pred_class_idx = get_prediction(text, model_name, emb_type)
            pred_class_label = label_encoder_classes[pred_class_idx]
            st.write(f'Predicted class: {pred_class_label}')
        elif 'regression' in model_name:
            pred_class_reg_norm = get_prediction(text, model_name, emb_type)
            pred_class_reg_raw = inverse_transform_prediction(pred_class_reg_norm)
            st.write(f'Predicted regression value: {pred_class_reg_raw}')
        else:
            st.write('Invalid model name')

run_app()
