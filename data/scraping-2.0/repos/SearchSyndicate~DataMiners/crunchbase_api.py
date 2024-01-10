#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:12:50 2023

@author: Kapil
"""

import requests
import json
import re
import pandas as pd
import time
from urls_info_retrive import domain_extract
from hugchat import hugchat
from semantic_search import semantic_search
from GPTturboAPI import openai_api
from serp_api import serp_response
import os
import unicodedata
from dotenv import load_dotenv


base_api_endpoint = "https://api.crunchbase.com/api/v4/"
# Load environment variables from .env file
load_dotenv()
you_api_key = os.environ.get('YOU_API_KEY')
crunchbase_api_key= os.environ.get('CRUNCH_BASE_API_KEY')

def get_headers():
    headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "X-cb-user-key": crunchbase_api_key
    }
    return headers

def removeSpecialChars(string):
    clean_string = re.sub(r"[^a-zA-Z0-9,'./:-]+", ' ', str(string)).strip()
    clean_string = clean_string.replace("key :","").replace(", value :",":").replace("/u","").replace('&amp;','')
    return clean_string

def removeAscendingChar(string):
    #converts Accented characters into english characters
    string=unicodedata.normalize('NFKD', str(string)).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return string

def name_comparison_score(name1,name2):

    name1_clean = removeSpecialChars(name1)
    name2_clean = removeSpecialChars(name2)
    name1_tokens = name1_clean.split(' ')
    name2_tokens = name2_clean.split(' ')
    length_1 = len(name1_tokens)
    length_2 = len(name2_tokens)
    score=0
    for i in name1_tokens:
        if i.lower() in [x.lower() for x in name2_tokens] and i not in '':
            score+=1
    return score/(length_1+length_2)

def get_main_company_name(company_name):
    """
      function to remove any abbreviations of legal entities types
    """
    with open('data/Entity_Legal_Names.txt') as f:
        entity_abbreviations = eval(f.readlines()[0])
    
    # Remove any entity type abbreviations from the company name
    company_words = company_name.split()
    main_words = []
    for word in company_words:
        word=word.replace(",","")
        #remove .com etc. from the names
        if "." in word:
            word = domain_extract(word)
        if word.lower() not in entity_abbreviations:
            main_words.append(word)
    main_company_name = " ".join(main_words)
    
    return main_company_name

def get_uuid(query, comp=True):
    """
    used for getting 1st returned uuid from API for a given company query
    and all descriptions of resembling companies upto 25
    Parameter: comp is there to decide whether to use name_comparison_score or not
    """
    response = requests.get(base_api_endpoint+
                            f"autocompletes?query={query}&collection_ids=organizations&limit=25",
                            headers=get_headers())
    similar_companies=None
    uuid='No results for given query'
    if response.status_code == 200:
        data = json.loads(response.text)
        uuid = data['entities'][0]['identifier']['uuid']
        if data['count']== 1:
            similar_companies = [(data['entities'][0]['identifier']['value'],
                                  data['entities'][0]['short_description'])]
        elif data['count']> 1:
            # removing dissimilar companies based on length and characters similarity
            if comp:
                similar_companies = [(i['identifier']['value'],i['short_description']) 
                                      for i in data['entities'] 
                                   if name_comparison_score(i['identifier']['value'],query)>0.15]
            else:
                similar_companies = [(i['identifier']['value'],i['short_description']) 
                                     for i in data['entities']]
    return similar_companies, uuid

def is_json(myjson):
  """
    function to check if a string is JSON
  """
  myjson = str(myjson).replace("'",'"')
  try:
    json.loads(myjson)
  except ValueError as e:
      print(str(e))
      return False
  return True



def prompting(prompt,company,semantic_urls, helper=False):
    """
    Prompts OpenAI 1st then in case of failure prompts Youchat
    and in case of failure in both, finally calls huggingchat 
    """
    url_link = ''
    output=None
    if helper:
        try:
            output, key_words = semantic_search(company, semantic_urls)
            print("semantic_search", output)
            
            if not re.search('[a-zA-Z]', output) or len(output)<60:
                output, url_link, _ = serp_response(company)
                print("serp_response", output)
                
        except:
            print("Crawler or serp couldn't extract any text")
            output, key_words = semantic_search(company, semantic_urls)
            print("semantic_search", output)
        
    if output==None and not helper:
        try:
            api="OpenAI 1"
            output = openai_api(prompt)
            print(type(output))
            print(api, output)
        except:
            api="Youchat"
            chat = f"{api} API down"
            response =requests.get(f"https://api.betterapi.net/youchat?inputs={prompt}&key={you_api_key}",
                               headers=get_headers())
            try:
                chat= json.loads(response.text)
            except:
                chat = {'generated_text':'error'}
            if response.status_code==200 or 'error' not in str(chat['generated_text']):
                if 'sorry' not in str(chat): 
                    output = chat['generated_text']
                    print(type(chat))
                    print(api, output)
                    try:
                        output = json.loads(re.search('({.+})', ' '.join(output.split('\n')).strip()).group(0).replace("'", '"'))
                    except:
                        print(f"JSON not returnd with {api}")
            else:
                output="{'Keywords':''}"
        if output!=None:
            keywords_len=0  
            try:  
                output = {k.lower(): v for k, v in eval(str(output)).items()}
                if is_json(output):
                    keywords_len = len(eval(str((output)))['keywords'].split(","))
            except:
                print(f"JSON not returnd with {api}")
            #check to prevent LLM Hallucinations
            if not len(output)>0 or keywords_len<4 or \
            not all(k in output for k in ("products","services","keywords")):
                try:
                    api="hugchat"
                    chatbot = hugchat.ChatBot(cookie_path="API_cookies/cookies.json")
                    # Create a new conversation
                    id = chatbot.new_conversation()
                    chatbot.change_conversation(id)
                    output = (chatbot.chat(json.dumps({"chat":prompt})))
                    print(type(output))
                    print(api, output)
                except:
                    error = "Hugchat down"
                    output = {'Products':error, 'Services':error}
        parsed_output = parse_llm_text(output)
        if parsed_output['Keywords']=='unknown' or len(parsed_output['Keywords'].split(","))<2:
            api="OpenAI 2"
            output_temp = openai_api(prompt)
            if "Quota exceeded" not in output_temp:
                output=output_temp
            print(type(output))
            print(api, output)
    return output, url_link

def get_products_from_text(text, company, country, semantic_urls):
    """
    Function to extract Product/Services from a text
    """
    company = removeSpecialChars(get_main_company_name(company))
    helper_prompt = f"""Give a list of the products and services offered by {company}. 
                        Limit your words to only relevant words. """
    sample, url_link = prompting(helper_prompt, company, semantic_urls, helper=True)
    def get_prompt(sample):
        if text!=None:
            prompt = f"""
            Your task is to help a marketing team to give useful informations
            about {company} from a given text.
            You have to perform the following actions: 
                
            1. Share the following informations about {company} with help of given text below:  
                - list of all Products sold by {company} across {country if country !="" else "the world"} separated by commas.
                - list of all Services offered by {company} across {country if country !="" else "the world"} separated by commas.
                - list of all Keywords about the Products or Services of the {company} separated by commas.
                  
            2. You must identify atleast one item either from Products or Services.
                There's no upper limit as long as they are relevant.
            
            3. Make your response as accurate as possible without any explanation or notes.
        
            4. Format your response only as one JSON object with 
                only "Products", "Services" and "Keywords" as the keys. 
                If the information isn't present in the test, use "unknown" as the value.
                
            text: '''{text}'''
            other helpful text: '''{sample}'''
            """
        else:
            sample_1, _, _ = serp_response(company)
            if sample in sample_1:
                sample_1 = openai_api(helper_prompt)
            prompt = f"""
            Your task is to help a marketing team to give useful informations
            about {company} from a given text.
            You have to perform the following actions: 
                
            1. Share the following informations about {company} with help of given text below: 
                - Extract a brief description about {company} in a sentence from the given text.
                - list of all Products sold by {company} across {country if country !="" else "the world"} separated by commas.
                - list of all Services offered by {company} across {country if country !="" else "the world"} separated by commas.
                - list of all Keywords about the Products or Services of the {company} separated by commas.
                  
            2. You must identify atleast one item either from Products or Services.
                There's no upper limit as long as they are relevant.
            
            3. Make your response as accurate as possible without any explanation or notes.
        
            4. Format your response only as one JSON object with 
                only "Description", "Products", "Services" and "Keywords" as the keys. 
                If the information isn't present in the test, use "unknown" as the value.
                
            text: '''{sample}'''
            other helpful text:'''{sample_1}'''
            """
        return prompt
    time.sleep(1)
    output, url_link = prompting(get_prompt(sample), company, semantic_urls)
    
    #taking care of edge case
    parsed_output = parse_llm_text(output)
    if ('unknown' in parsed_output['Services'].lower()) \
        and ('unknown' in parsed_output['Products'].lower()):
            print("taking care of edge case")
            text=None
            sample, url_link, _ = serp_response(company)
            print(sample)
            output, url_link = prompting(get_prompt(sample), company, semantic_urls)   
            print(output, "final_output")
    return output, url_link

def parse_llm_text(string):
    output={}
    string = removeAscendingChar(string)
    string = removeSpecialChars(string)
    products_str=services_str=keywords_str=''
    products_match = re.search("products :", string.lower())
    services_match = re.search("services :", string.lower())
    keywords_match = re.search("keywords :", string.lower())
    
    if not products_match:
        products_match = re.search("Products", string)
    if not products_match:
        products_match = re.search("Product", string)
    if not products_match:
        products_match = re.search("'products':", string)
        
    if not services_match:
        services_match = re.search("Services", string)
    if not services_match:
        services_match = re.search("Service", string)
    if not services_match:
        services_match = re.search("'services':", string)
        
    if not keywords_match:
        keywords_match = re.search("Keywords", string)
    if not keywords_match:
        keywords_match = re.search("Keyword", string)
    if not keywords_match:
        keywords_match = re.search("'keywords':", string)
        
    if products_match and services_match and keywords_match:
        products_str = string[products_match.end():services_match.start()].lstrip().rstrip()
        services_str = string[services_match.end():keywords_match.start()].lstrip().rstrip()
        keywords_str = string[keywords_match.end():].lstrip().rstrip()
    
    if "description" in string.lower():
        description_match = re.search("description :", string.lower())
        if not description_match:
            description_match = re.search("Description", string)
        if not description_match:
            description_match = re.search("description", string)
        if not description_match:
            description_match = re.search("'description':", string.lower())
        if description_match:
            description_str = string[description_match.end():products_match.start()].lstrip().rstrip()
            output["Description"] = description_str.rstrip(", ") if description_str !="" else "unknown"
    output["Products"] = products_str.rstrip(", ") if products_str !="" else "unknown"
    output["Services"] = services_str.rstrip(", ") if services_str !="" else "unknown"
    output["Keywords"] = keywords_str.rstrip(", ") if keywords_str !="" else "unknown"
    return output

def batch_call_api(df):
    """
    Function to call crunchbase api 25 times in every minute 
    max limit is 25
    """
    errors=[]
    desc_df=pd.DataFrame(columns=['company', 'description'])
    for index, row in df.iterrows():
        # Extract the query with company name and country from the DataFrame
        query = input_df.loc[index,'queries']
        if index % 25 == 0:
            # Create the API request 
            similar_companies, uuid = get_uuid(query, comp=False)
    
            # Check if the response was successful
            if similar_companies:
                # Extract the descriptions the API response
                desc_df = pd.concat([desc_df,
                          pd.DataFrame(similar_companies, columns=['company', 'description'])],
                          ignore_index=True)
                if uuid!='No results for given query':
                    df.loc[index,'uuid'] = uuid
                else:
                    df.loc[index,'uuid'] = ''
            else:
                # Add the company to the error list
                errors.append(query)
            print(f"Taking a minute break...{index/25}")
            time.sleep(60)
    
        # Check if all inputs have been given
        if index == len(df) - 1:
            print("All inputs have been processed.")
            return errors, df,desc_df


if __name__ == '__main__':
    batch_calling=False
    
    if batch_calling: 
        input_df = pd.read_csv("data/unicorn-company-list.csv",keep_default_na=False)
        input_df['queries']=input_df.apply(lambda x: x['Company']+", "+x['Country'],axis=1)
        errors, df,desc_df = batch_call_api(input_df)
    
    api_query = "Amazon.com, Inc., USA"
    similar_companies, uuid = get_uuid(api_query)
    print(similar_companies)
