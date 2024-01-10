from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import torch
import torch
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
import numpy as np
import torch
import transformers as ppb 


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS =nltk.corpus.stopwords.words('english')

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text







from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')





def gpt_fscore(text,model,tokenizer):
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0) 
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    return last_hidden_states




fn=['dataset']
for k in range(0,1):
    fname='D:/data/aman_api_postmatem/'+fn[k]+'.csv'
    df = pd.read_csv(fname,encoding='latin-1')
    df['quote'] = df['quote'].apply(clean_text)
    print(df.head(10))
    train_tokens = list(map(lambda t:   tokenizer.tokenize(t)+['[CLS]'], df['quote']))
    train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids,train_tokens))
    b=np.zeros((np.shape(df)[0]))
    for i in range(0,np.shape(df)[0]):
        b[i]=np.shape(train_tokens_ids[i])[0]
    pad=np.zeros((np.shape(df)[0],int(np.max(b))))
    for i in range(0,np.shape(df)[0]):
        a=train_tokens_ids[i]
        for j in range(0,np.shape(a)[0]):
            pad[i,j]=a[j]
    
    pad1=pad[1:500,:]       
    input_ids = torch.tensor(np.array(pad1))        
    with torch.no_grad():
        last_hidden_states = model(input_ids.long())    
    features = last_hidden_states[0][:,0,:].numpy()
    fname='D:/data/aman_api_postmatem/gpt2'+fn[k]+'.csv'
    np.savetxt(fname,features, delimiter=',', fmt='%f')     
#df = pd.read_csv('D:/data/msr2013-bug_dataset-master/PDE.csv',encoding='latin-1')
#df['Var3'] = df['Var3'].apply(clean_text)
#print(df.head(10))
#train_tokens = list(map(lambda t:   tokenizer.tokenize(t)+['[CLS]'], df['Var3']))
#train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids,train_tokens))

#b=np.zeros((np.shape(df)[0]))
#for i in range(0,np.shape(df)[0]):
 #   b[i]=np.shape(train_tokens_ids[i])[0]
    
    
#pad=np.zeros((np.shape(df)[0],int(np.max(b))))
#for i in range(0,np.shape(df)[0]):
 #   a=train_tokens_ids[i]
  #  for j in range(0,np.shape(a)[0]):
   #     pad[i,j]=a[j]
        

pad1=pad[200:400,:]       
input_ids = torch.tensor(np.array(pad1))        
with torch.no_grad():
    last_hidden_states = model(input_ids.long())    
features = last_hidden_states[0][:,0,:].numpy()
fname='D:/data/aman_api_postmatem/gpt21.csv'
np.savetxt(fname,features, delimiter=',', fmt='%f') 