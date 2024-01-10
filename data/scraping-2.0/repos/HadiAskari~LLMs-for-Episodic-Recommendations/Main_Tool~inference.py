import os
import openai
import pandas as pd
import tiktoken
import json
import requests
import pickle as pkl
from collections import defaultdict
from googlesearch import search
from fake_useragent import UserAgent
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import re

def num_tokens_from_messages(messages):
    encoding= tiktoken.get_encoding("cl100k_base")  #model to encoding mapping https://github.com/openai/tiktoken/blob/main/tiktoken/model.py
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        # print(message)
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

# def get_synopsis(series, episode, MAFContext):
    
#     ua = UserAgent()
#     query = "site:rottentomatoes.com {} {}".format(series, episode)

#     print(query)

#     count = 0
#     while count < 5:
#         try:
#             results = search(query, num=10, user_agent=ua.random)
#             url = next(results)
#             break
#         except Exception as e:
#             print(e)
#             count += 1
    
#     if count >= 5:
#         return MAFContext

#     #call first url
#     r=requests.get(url)
    
#     text=r.text
#     print(url)
#     #extract soup
#     soup=BeautifulSoup(text, 'html.parser')

#     #extract relevant data
#     synopsis_elem=soup.find('drawer-more')

#     if not synopsis_elem:
#         print("Rotten Tomatoes Not Found")
#         return MAFContext
    
#     return synopsis_elem.get_text()

def prompt(Tier1, series, episode, context):
    if episode==None:
        episode='Not available'
    
    Tier1=" \n".join(Tier1)

    prompt="""Based of the following summary/synopsis of the episode titled: {}
from the series titled {}:

{}

Which of the following Tiers are relevant ad categories for this episode along with
a description of why they are relevant and a similarity score with 0 being no 
relevance and 1 meaning completely relevant? 

{}

Return your response in the following format:

1. Tier Name - Relevance Score () "\n" Description. "\n\n"

2. Tier Name - Relevance Score () "\n" Description. "\n\n" ...

""".format(episode,series,context,Tier1)

    return prompt

def prompt_output_parse(output):
    #relevant adwords to have key as adword and dict of 2 values as description and score
    relevant_adwords={}
    first_split=output.split('\n\n')
    # print(len(first_split))
    
    if len(first_split)==0:
        return None
    
    for splits in first_split:
        temp={}
        # print(splits)
        mid_split=splits.split('\n')
        try:
            res = re.findall(r'\(.*?\)', mid_split[0])[0]
            res=res.replace('(', "")
            res=res.replace(')', "")
        except:
            print(mid_split)
            temp['Description']='error'
            temp['Relevance Score']='error'
            relevant_adwords['error']=temp
            continue


        
        #print(float(res[0]))
        
        res=float(res)
        
        if res<0.1:
            continue
        
        adword=mid_split[0].split('-')[0].split('.')[1].strip()
        number=res
        try:
            description=mid_split[1]
            temp['Description']=description
            temp['Relevance Score']=number
            
        except:
            print('Split error')
            print(splits)
            temp['Description']='error'
            temp['Relevance Score']='error'

        relevant_adwords[adword]=temp



    return relevant_adwords


def deployment_stuff():
    with open('azure-configuration.json') as inputfile:
        azureconfig = json.load(inputfile)
    openai.api_key = azureconfig['key'] 
    openai.api_base = azureconfig['endpoint'] 
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15' # this may change in the future

    deployment_name= azureconfig['deployment_name']

    return deployment_name

def call(deployment_name, series, episode, context, Tier1, initial_dic):
    #Right now calling requests to get the synopsis, once Dataset is ready, replace it
    #context=get_synopsis(series, episode, context)

    #OpenAI Stuff
    system_message={"role": "system", "content": "You are a helpful assistant."}
    max_response_tokens = 4096
    token_limit = 2000
    conversation = []
    conversation.append(system_message)

    final_res=[]

    
    user_input = prompt(Tier1, series, episode, context)
    
    conversation.append({"role": "user", "content": user_input})
    conv_history_tokens = num_tokens_from_messages(conversation)
    # print(conv_history_tokens)

    # while conv_history_tokens + max_response_tokens >= token_limit:
    #     del conversation[0] 
    #     conv_history_tokens = num_tokens_from_messages(conversation)

    response = openai.ChatCompletion.create(
        engine=deployment_name, # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
        messages=conversation,
        temperature=0.7,
        max_tokens=max_response_tokens,
    )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    # print(conversation)

    responses=prompt_output_parse(response['choices'][0]['message']['content'])

    # print(responses)

    final_res.append(responses)

    if not responses:
        #return empty
        return None
    
    #2nd Call

    for adwords in responses.keys():

        #print(adwords)
        try:
            Tiers=initial_dic[adwords]["Children"]
        except:
            print(adwords)
            continue
        if not Tiers:
            continue
        
        Tiers=" \n".join(Tiers)

        # print(Tiers)

        user_input = prompt(Tiers, series, episode, context)

        conversation.append({"role": "user", "content": user_input})
        conv_history_tokens = num_tokens_from_messages(conversation)
        # print(conv_history_tokens)

        response_new = openai.ChatCompletion.create(
        engine=deployment_name, # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
        messages=conversation,
        temperature=0.7,
        max_tokens=max_response_tokens,
        )

        conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
        # print(conversation)

        responses_new=prompt_output_parse(response_new['choices'][0]['message']['content'])

        # print(responses)

        final_res.append(responses_new)

        if not responses_new:
            #return empty
            return None


    return final_res

def pre_reqs():
    with open('data/List_of_IAB.pkl', 'rb') as f:
        IAB_list=pkl.load(f)

    with open('data/initial_KG_dic.pkl', 'rb') as f:
        initial_dic=pkl.load(f)

    dfoldmappings=pd.read_csv('data/MAFlabel_to_Adword_mappings_202210_combined_mappings.csv')
    listofadwords=dfoldmappings['adword'].to_list()
    listofadwords=[x.split('|') for x in listofadwords]
    for id, x in enumerate(listofadwords):
        try:
            x.remove('')
            listofadwords[id]=x
        except:
            listofadwords[id]=x

    listofadwords=[list(item) for item in set(tuple(row) for row in listofadwords)]

    Tier1=[]
    for items in listofadwords:
        if len(items)==1 and items not in Tier1:
            Tier1.append(items[0])

    return Tier1, initial_dic

def main(series_titles, episode_titles, medium_synopsiss):
    #df_data=pd.read_csv('Rotten_Tomatoes_Synopsis.csv')
    
    Tier1, initial_dic=pre_reqs()

    # return 'abcd'


    series_titles=series_titles
    episode_titles=episode_titles
    medium_synopsiss=medium_synopsiss

    dep_name=deployment_stuff()
    # print(dep_name)
    
    # print(series_title[0])
    # print(episode_title[0])

    series=[]
    episode=[]
    synopsis=[]
    iab_label=[]
    relevance_clue=[]
    relevance_score=[]

    
    for series_title,episode_title,medium_synopsis in tqdm(zip(series_titles,episode_titles,medium_synopsiss), total=len(series_titles)):
        try:
        
            final_res=call(dep_name,series_title,episode_title,medium_synopsis,Tier1,initial_dic)
            if not final_res:
                continue
            for idx in range(len(final_res)):
                for k in final_res[idx].keys():
                    iab_label.append(k)
                    relevance_clue.append(final_res[idx][k]['Description'])
                    relevance_score.append(final_res[idx][k]['Relevance Score'])
                    series.append(series_title)   
                    episode.append(episode_title)
                    synopsis.append(medium_synopsis)
                
        except Exception as e:
            print(e)
            continue

    final_dic={"Series Title":series, "Episode Title": episode, "RT Synopsis": synopsis, "Predicted IAB Label":  iab_label, "Relevance Clue": relevance_clue, "Relevance Score": relevance_score}

    return final_dic