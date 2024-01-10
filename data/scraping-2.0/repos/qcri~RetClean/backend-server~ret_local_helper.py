import torch
import openai
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

matching_pipeline = pipeline('text-classification', model='shamz15531/roberta_tuple_matcher_base')

### Converts serialized tuple into Matcher Format Serialized Tuple
def convert_to_special_tokenization(t1, mode):
    t1 = " ".join(t1.split()[:100]) # Hard Coded
    t2 = t1.replace("]","").replace("[","[ATT]").replace(":", "[VAL]").replace(";","[ATT]")
    if mode == "q":
        t2 = t2 + "[MS]"
    else:
        t2 = t2
    return t2
    
# Makes pairs of Matcher Format Serialized Tuples. Query TUple + Each Retrieved Tuple from list
def make_pairs(query_tuple, retrieved_list):
    ret_pairs = []
    converted_query = convert_to_special_tokenization(query_tuple, "q")
    
    for i in range(len(retrieved_list)):
        temp = convert_to_special_tokenization(retrieved_list[i], None) + " [SEP] " + converted_query + " [SEP] "
        temp = temp.replace('  ', ' ').strip()
        ret_pairs.append(temp)
        
    return ret_pairs

def matching(all_retrieved, # format = return by search_index()
             all_query_tuples, # str serialized,
             model_directory = 'shamz15531/roberta_tuple_matcher_base',
             matching_pipeline = matching_pipeline
             ):
    

    # Create Pipeline if None
    if matching_pipeline == None or model_directory!='shamz15531/roberta_tuple_matcher_base':
        matching_pipeline = pipeline('text-classification', model=model_directory)

    if len(all_retrieved) != len(all_query_tuples):
        raise ValueError("Number of Query Tuples and Retrieved Sets does not match")
    
    all_matched_sets = []

    for i in range(len(all_query_tuples)):
        query_tuple = all_query_tuples[i]
        retrieved = all_retrieved[i]

        all_match_for_this_query = []

        # Make Pairs
        retrieved_tuples = retrieved["serialization"]
        pairs = make_pairs(query_tuple, retrieved_tuples)
        
        # Filtered Matched Dictionary
        ret_matched = {"serialization" : "",
                    "table" : "",
                    "index" : "",
                    "org_retrieved_index" : ""
                    }
        
        matched_1 = False # for now we only care about the top match, this can be removed if more than 1 repair it so be returned

        # Do Matching
        for i in range(len(pairs)):
            if matched_1 == False: # Stop once 1 match is found
                pred = matching_pipeline(pairs[i])
                if pred[0]["label"] == "LABEL_1":
                    ret_matched["serialization"] = retrieved["serialization"][i]
                    ret_matched["table"] = retrieved["table"][i]
                    ret_matched["index"]= retrieved["index"][i]
                    ret_matched["org_retrieved_index"] = i
                    matched_1 = True
        # Return Matches
        all_matched_sets.append(ret_matched)
    return all_matched_sets

# Load Reasoner (Extractor)
def load_reasoner(mode = "ST"):
    if mode == 'ST':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    elif mode == 'GPT':
        service_name = "dataprepopenai"
        deployment_name = "gpt3_davinci_imputer"
        key = "c4abc69022d84e04aacdda594b3ba273"  # please replace this with your key as a string
        openai.api_key = key
        openai.api_base = "https://{}.openai.azure.com/".format(service_name) # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
        openai.api_type = 'azure'
        openai.api_version = '2022-06-01-preview' # this may change in the future
        deployment_id=deployment_name #This will correspond to the custom name you chose for your deployment when you deployed a model.
        return deployment_id
    
# GPT Extractor Prompt Function
def generate(context, impute_attribute, engine, max_tokens):
    prompt = f'Given {context} what is the value of the attribute closest to {impute_attribute}?'
    response = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_tokens)
    return response

# Helper function for extraction
def scan(impute_att, context, model):
    att_val_pairs = [x.strip().split(' : ') for x in context[2:-2].split(' ; ')]
    if att_val_pairs == [['']] or context == "":
        return ""
    
    atts = [x[0] for x in att_val_pairs]
    vals = [x[1] for x in att_val_pairs]
    impute_atts = [impute_att for i in range(len(atts))]
    
    embeddings1 = model.encode(impute_atts, convert_to_tensor=True)
    embeddings2 = model.encode(atts, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    pairs = []
    for i in range(len(cosine_scores)):
        pairs.append({
            'attribute': atts[i],
            'value': vals[i],
            'score': cosine_scores[0][i]
        })

    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    return pairs[0]['value'] 

# Extraction Function
def extraction(list_ret_tuples,
               impute_attribute, # <str> impute attribute name
               query_tuple = None, # maybe needed for later
               mode = "ST", # ST for 'SentenceTransformers' or GPT for GPT3/3.5
               limit = None,
               reasoner_model = None
               ):
    # Load reasoner. Sentence Transformer by Default
    if reasoner_model == None:
        loaded_reasoner = load_reasoner(mode)
    else:
        loaded_reasoner = reasoner_model
    
    # Limit the 
    if limit!=None:
        list_ret_tuples['serialization'] = list_ret_tuples['serialization'][:limit]
        list_ret_tuples['table'] = list_ret_tuples['table'][:limit]
        list_ret_tuples['index'] = list_ret_tuples['index'][:limit]
        
    results = []

    context_tuple = list_ret_tuples['serialization']
    context_table = list_ret_tuples['table']
    context_index = list_ret_tuples['index']

    if mode == 'ST':
        val = scan(impute_attribute, context_tuple, loaded_reasoner)
    elif mode == 'GPT':
        val = generate(impute_attribute, context_tuple, loaded_reasoner)

    obj = {
        'source' : context_tuple,
        'repair': val,
        'table': context_table,
        'index': context_index
    }
    results.append(obj)

    return results
  

print("LOADED ret_local_helper")
            
    

