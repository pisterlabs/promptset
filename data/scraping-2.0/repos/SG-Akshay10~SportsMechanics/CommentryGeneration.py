#!/usr/bin/env python
# coding: utf-8

# # Commentry Generation

# In[113]:


import numpy as np
import pandas as pd
import tensorflow as tf
import random


# In[114]:


#!pip install openai
#!python -m pip install cohere


# In[115]:


import cohere

co = cohere.Client('JY6As4HcMbxbXz6QJegvMAfNWflhvT0CoNxeN8pZ')


# In[116]:


df = pd.read_csv(r"deliveries.csv")


# In[117]:


df1 = df[df['match_id'].between(1, 20)]


# In[118]:


selected_columns = ['batsman', 'bowler', 'total_runs']
df2 = df[selected_columns]


# In[129]:


commentaries = []
i = random.randint(0,20) 
print(df2.batsman[i],df2.bowler[i],df2.total_runs[i])
response = co.generate(
    prompt=f"Generate a commentary of the following cricket ball in Ravi Shastri style make sure runs scored are the important parameter: Batsman: {df2.batsman[i]}, Bowler: {df2.bowler[i]}, runs scored: 6. Make sure it is around 30 words length",
)
commentaries.append(response[0].text)


# In[130]:


for i in range(len(commentaries)):
    print(commentaries[0])
    print("\n")

