import random
import json
import yaml
import torch
import discord
import os
from pyllamacpp.model import Model
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from intent_actions import check_intent
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI


with open('intents.json', 'r') as f:
    intents = json.load(f)
with open('secure/secrets.yml', 'r') as f:
    data = yaml.safe_load(f)
    TOKEN = data['token']
    ADMIN = data['bot_admin']
    OPENAI_KEY = data['openAI']
    
os.environ['OPENAI_API_KEY'] = OPENAI_KEY


loader = DirectoryLoader('data')
index = VectorstoreIndexCreator().from_loaders([loader])

def check_author(message):
    if message.author == client.user and str(message.author) != ADMIN:
      return False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
client = discord.Client(intents=discord.Intents.default())



FILE = "models/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Edith"
print("Let's chat! (type 'quit' to exit)")
@client.event
async def on_ready():
  print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
  if message.author == client.user:
    return
  mgs = str(message.content)
  # if check_author(message) == False:
  #   return
  if not '#' in mgs and not '/' in mgs:
    response = index.query(mgs, llm=ChatOpenAI())
    await message.channel.send(response)
    return
    
  mgs = mgs[1:]
  sentence = mgs
  sentence = tokenize(sentence)
  X = bag_of_words(sentence, all_words)
  X = X.reshape(1, X.shape[0])
  X = torch.from_numpy(X).to(device)   
  output = model(X)
  _, predicted = torch.max(output, dim=1)  
  tag = tags[predicted.item()]     
  probs = torch.softmax(output, dim=1)
  prob = probs[0][predicted.item()]
  if prob.item() > 0.75:
      for intent in intents['intents']:
          if tag == intent["tag"]:
              check_intent(intent["tag"], mgs)
              await message.channel.send(f"{random.choice(intent['responses'])}")
              print(f'tag: {intent["tag"]}')
  else:
    response = index.query(mgs, llm=ChatOpenAI())
    await message.channel.send(response)
    

client.run(TOKEN)