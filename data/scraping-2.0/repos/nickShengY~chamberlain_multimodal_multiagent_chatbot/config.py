import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from openai import OpenAI
import os, re
from langchain.chat_models import ChatOpenAI
import datetime
from api_key import api_key
# OpenAI TTS Agent
t2sp = OpenAI()
# Constants for the number of labels and tokenizer
NUM_LABELS = 14
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-base')

# Defining labels for different categories and Yes/No responses
LABELS = ['Eat', 'Dress', 'Bill', 'Finance', 'Planner', 'Grocery', 'Laundry', 'Fun', 'IoT', 'Shopping', 'Flight', 'Coding', 'Task', 'Other']
YNLABLES = ['No','Yes']

# Mapping labels to numerical indices
LABELS = {label: idx for idx, label in enumerate(LABELS)}
YNLABLES = {label: idx for idx, label in enumerate(YNLABLES)}

# Setting up the device for model computation (GPU or CPU)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Loading and preparing the question classification model
QMODEL = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=NUM_LABELS)
QMODEL.load_state_dict(torch.load('question_model.pt'))
QMODEL.to(DEVICE)

VMODEL = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
VMODEL.load_state_dict(torch.load('label_model.pt'))
VMODEL.to(DEVICE)

# Initialize various OpenAI clients for different tasks

embedding_size = 1536 # Set the size of embeddings used in the models

question_judge = OpenAI(
  api_key=api_key,
)


chat_bot = OpenAI(
  api_key=api_key,
)

grocery_client = OpenAI(
  api_key=api_key,
)

cook_bot = OpenAI(
  api_key=api_key,
)
client_helper = OpenAI(
    api_key=api_key,
)
cook_helper = OpenAI(
    api_key=api_key,
)


# Dir for the data
eatdir = '' # my eating/cooking plans, should be an excel
bills_dir = 'D:/Projects/DeepL/NLP_app/data/bills'
receipts = 'D:/Projects/DeepL/NLP_app/data/reciepts' # all my receipts, should be an excel
dailyplan = '' # my daily plans, should be an excel
fridge = 'D:/Projects/DeepL/NLP_app/data/fridge.csv' # my fridge, should be a csv
laundry_basket = '' # dirty landry records, should be an excel
Iot_control = '' # No need now
shopping_dir = 'D:/Projects/DeepL/NLP_app/data/items/shopping_cart' # shopping cart dir, should connect to my items
item_dir = 'D:/Projects/DeepL/NLP_app/data/items' # items dir
selfie_dir = 'D:/Projects/DeepL/NLP_app/data/photos' # dress
code_dev_dir = 'D:/Projects/DeepL/NLP_app/data/playground' # code generation root dir, no imple now
Finance_port_dir = 'D:/Projects/DeepL/NLP_app/data/finance' # Fin
chat_history = ''


now = datetime.datetime.now()
# Format the datetime in the desired format (Year-Month-Day-Hour-Minute-Second)
formatted_datetime = now.strftime("%Y-%m-%d-%H-%M-%S")