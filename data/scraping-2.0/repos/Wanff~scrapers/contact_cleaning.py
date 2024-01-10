#%%
import os
import pandas as pd
import numpy as np
import time
from openai import OpenAI
import pickle 

from dotenv import load_dotenv

# %%
load_dotenv()

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

class OpenAIModel():
    def __init__(self, engine, system_prompt = None):
        self.engine = engine
        self.system_prompt = system_prompt
    
    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return client.embeddings.create(
                input = [text], 
                model=self.engine)['data'][0]['embedding']
    
    def get_chat_completion(self, messages, max_tokens: int = 1700):
        return client.chat.completions.create(
            model=self.engine,
            messages=messages,
            max_tokens = max_tokens,
            )['choices'][0]['message']['content']

    def classify_text(self, query):        
        return client.chat.completions.create(
        model=self.engine,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        )

#%%
def is_actual_founder(bio, model):
    query = f"Bio: {bio}\nYou:"
    response_json = eval(model.classify_text(query).choices[0].message.content)
    try:
        return response_json['founder'] == 1
    except:
        return False
#%%
#* twitter cleans

system_prompt = 'You are an intelligent and helpful assistant, who given a Twitter bio, will output a JSON file that indicates whether the given person is a young startup founder or not. People who are investors/VCs, were previously founders, or professors/researchers should not count. Output should look like {"founder" : 1} or {"founder" : 0}. Here are some examples:\nBio: Peter Dolch,Entrepreneur | Founder | Investor | Advisor | MIT | Sloan | Founder Annual MIT Female Founders Pitch Competition | MIT Sloan Club of NY Executive Board,MIT.\nYou: {"founder":0}\nBio:Founder of @joinwarp | @MIT \'20 | 🇮🇳 🇺🇸,MIT\nYou:{"founder":0}'
gpt4 = OpenAIModel("gpt-4-1106-preview", system_prompt = system_prompt)
response = gpt4.classify_text("Bio: Founder of @Algorand and co-inventor of zero-knowledge proofs. @MIT Professor. Accademia Dei Lincei. This is my only official social media account.,MIT\nYou:")

print(response.choices[0].message.content)
    
#%%

twitter_df = pd.read_csv("final_nitter_scrapes.csv", usecols=['username','name','bio','school'])

founders = pd.DataFrame(columns=['username','name','bio','school'])
for i, person in twitter_df.iterrows():
    bio = person['bio']
    if is_actual_founder(bio):
        founders = pd.concat([pd.DataFrame(person).T, founders], ignore_index=True)

founders
# %%
for i, person in list(twitter_df.iterrows())[703:]:
    bio = person['bio']
    print(person['name'])
    print(bio)
    if is_actual_founder(bio):
        print("FOUNDER")
        founders = pd.concat([pd.DataFrame(person).T, founders], ignore_index=True)
    print()

#%%
founders['username'] = founders['username'].apply(lambda x: x.replace('https://nitter.net/', 'https://twitter.com/'))
#%%
founders.to_csv("twitter_founders.csv")
# %%
#* linkedin cleans
system_prompt = 'You are an intelligent and helpful assistant, who given a LinkedIn subtitle, will output a JSON file that indicates whether the given person is a young startup founder or not. People who are investors/VCs, were previously founders, are founders of non profits or professors/researchers should not count. Additionally, if the person is clearly not affiliated with Harvard, MIT or Stanford, they should not count. Output should look like {"founder" : 1} or {"founder" : 0}. Here are some examples:\nSubtitle: CS at Harvard | Founder of Harvard Tech for Social Good,Harvard\nYou: {"founder":0}\nSubtitle:Founder at Stealth | Harvard CS,Harvard\nYou:{"founder":0\nSubtitle: ,Econ @ Harvard | Music @ Berklee,Harvard\nYou:{"founder":0}\nSubtitle: https://www.linkedin.com/in/rexwoodbury\nYou:{"founder":0}'

gpt4 = OpenAIModel("gpt-4-1106-preview", system_prompt = system_prompt)
# %%
harvard = pd.read_csv("harvard_linkedin.csv", usecols=['name','linkedin_url','subtitle','school'])
stanford = pd.read_csv("stanford_linkedin.csv", usecols=['name','linkedin_url','subtitle','school'])

linkedin_df = pd.concat([harvard, stanford], ignore_index=True)
linkedin_df

#%%
founders = pd.DataFrame(columns=['name','linkedin_url','subtitle','school'])
for i, person in linkedin_df.iterrows():
    subtitle = person['subtitle']
    print(person['name'])
    print(person['subtitle'])
    if is_actual_founder(subtitle, gpt4):
        print("FOUNDER")
        founders = pd.concat([pd.DataFrame(person).T, founders], ignore_index=True)
    print()
# %%
founders.to_csv("linkedin_founders.csv")
# %%
#* merging databases

twitter_founders = pd.read_csv("twitter_founders.csv", usecols=['username','name','bio','school'])
linkedin_founders = pd.read_csv("linkedin_founders.csv", usecols=['name','linkedin_url','subtitle','school'])
twitter_founders.columns = ['twitter', 'name', 'twitter_bio', 'school']
linkedin_founders.columns = ['name', 'linkedin', 'subtitle', 'school']
#%%
twitter_founders['name'] = twitter_founders['name'].apply(lambda x: x.replace("'", ""))
# %%
yc_founders = pickle.load(open("yc_founders.pickle", "rb"))

yc_founders_df = pd.DataFrame(yc_founders, columns=['company_name', 'founder_name', 'founder_linkedin', 'founder_twitter', 'founder_bio', 'yc_link', 'school'])

yc_founders_df.columns = ['company_name', 'name', 'linkedin', 'twitter', 'desc', 'yc_link', 'school']
yc_founders_df

# %%
founders = pd.concat([yc_founders_df, linkedin_founders, twitter_founders], ignore_index=True)

founders
#%%
founders.drop_duplicates(inplace=True)
founders

#%%
founders.drop_duplicates(subset=['name'], keep='last', inplace=True)
founders
# %%
founders.dropna(subset=['twitter', 'linkedin'], how='all', inplace=True)
founders
# %%
founders['reachout_link'] = founders['linkedin'].fillna(founders['twitter'])
founders
# %%
founders.to_csv("scraped_founders.csv")

# %%
