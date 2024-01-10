import urllib.request
import json
import os
import argparse
from pathlib import Path
import openai
import re
import time


parser = argparse.ArgumentParser()
parser.add_argument("--file_path_in", type=str)
parser.add_argument("--gpt_engine", type=str)
parser.add_argument("--temperature_clasify", type=float, default=0.9)
parser.add_argument("--max_tokens_clasify", type=int, default=100)
parser.add_argument("--temperature_summarize", type=float, default=0.9)
parser.add_argument("--max_token_summarize", type=int, default=500)

parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--frequency_penalty", type=float, default=0.5)
parser.add_argument("--presence_penalty", type=float, default=0.5)
parser.add_argument("--best_of", type=float, default=1)

parser.add_argument('--file_path_out', type=str)


openai.api_type = "azure"
openai.api_key = '<Add OpenAI Credential>'
openai.api_base = '<Add OpenAI Base API>'
openai.api_version = "2022-12-01"
entity_types = ["Research Orgnization","Technology","Tobacco" , "Media"]


args = parser.parse_args()

def splitter(n, s):
    pieces = s.split()
    list_out = [" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n)]
    return list_out

def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

def trim_incomplete(t):
    if t.endswith('.'):
        if not re.search('[a-z]\.$',t):
            t = t[:-1]

    if not t.endswith('.'):
        t = t.rsplit('. ', 1)[:-1]
        t = "".join(t)+'.'
    
    t = t.strip()
    return t


def recognize_categories(prompt, engine,  temperature, max_tokens):
    """Recognize entities in text using OpenAI's text classification API."""
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].text



def summarize_information(prompt, engine, temperature , max_tokens , top_p , frequency_penalty , presence_penalty , best_of):
     
    response = openai.Completion.create(
        engine= engine,
        prompt = prompt,
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty =presence_penalty,
        stop=['\n\n###\n\n'], #the ending token used during inference, once reaches this token GPT-3 knows the completion is over
        best_of = best_of
    )
    return response.choices[0].text



output_path = f"{Path(args.file_path_out)}/categorized_json/"
print(f"output_path{output_path}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

path = f"{Path(args.file_path_in)}/raw_json/"

files = os.listdir(path)


print(f"temperature_clasify:{args.gpt_engine}")
print(f"max_tokens_clasify:{args.temperature_clasify}")
print(f"frequency_penalty:{args.frequency_penalty}")

#engine = args.gpt_engine
## temp ###
engine = 'davinci'

for fileName in files:
        print(f"{path}{fileName}")
        with open(f"{path}{fileName}", 'rb') as f2: 
            json_results = json.load(f2)
        
        for idx, invoice in enumerate(json_results['documents']): 
            vendorName = invoice.get("VendorName")
            


            #### Summarize vendor information 
            try:
                
                prompt_i = 'Summarize the vendor introduction, industry and products given the vendor name.\n\nvendor name:\n'+normalize_text(str(vendorName))+'\n\nSummary:\n'    
                print(prompt_i)
                #vendor_info = normalize_text(summarize_information(normalize_text(prompt_i), engine=engine, temperature=args.temperature_summarize, max_tokens=args.max_token_summarize, top_p=args.top_p, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty, best_of=args.best_of))       
                vendor_info = ""
                #time.sleep(6)
                    
            except:
                vendor_info = ""

            #### Vendor Categorization
             
            if vendor_info =="":
                category = 'Not Identified'
            else:
                try:

                    prompt = f"Classify the following vendor information into one of the following categories: : {', '.join(entity_types)} \n\n vendor information:{vendor_info}\n\n Classified category:"
                    print(prompt)
                    #category = normalize_text(recognize_categories(prompt, engine=engine,  temperature=args.temperature_clasify, max_tokens=args.max_tokens_clasify))
                    category = 'Not Identified'
                except:
                    category = 'Not Identified'
            print(f"{category}:{vendor_info}")

            for idx2, product in enumerate(invoice['Items']): 
                Description = product.get("Description")
                prompt_i = 'Summarize the product information given the product name and its company name \n\n\product name:\n'+normalize_text(str(Description))+ '\n\company name:\n'+normalize_text(str(vendorName))+'\n\nSummary:\n'
                print(prompt_i)
                try:
                    product_info = normalize_text(summarize_information(normalize_text(prompt_i), engine=engine, temperature=args.temperature_summarize, max_tokens=args.max_token_summarize, top_p=args.top_p, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty, best_of=args.best_of))
                    #time.sleep(6)
                except:
                    product_info = ''

                print(f"product_info:{Description}:{product_info}")
                invoice['Items'][idx2]['Product_Summarization'] = product_info

            json_results['documents'][idx]['Data_Category'] = category
            json_results['documents'][idx]['Data_Summarization'] = vendor_info
             


        with open(f"{output_path}{fileName}", 'w') as f3: 
            json.dump(json_results, f3)

print(f"OpenAI data categorzation process complete")