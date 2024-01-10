import pandas as pd
import numpy as np
import torch
import openai
import time
import torch
from dotenv import load_dotenv
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_dotenv()
    
# Load Post Processing Models
roberta_qa_model = AutoModelForQuestionAnswering.from_pretrained("shamz15531/roberta_repair_extractor")
roberta_qa_tokenizer = AutoTokenizer.from_pretrained("shamz15531/roberta_repair_extractor")

def custom_prompt_convert(json_data, custom_prompt, impute_att):
    df = recieved_json_to_pdf(json_data)
    prompt_words = custom_prompt.split(' ')
    final_prompt_list = []
    for i in range(df.shape[0]):
        row = df.iloc[i,:]    
        ret_prompt_words = []
        for word in prompt_words:
            if word[0] == "$" and word[-1] == "$":
                ret_prompt_words.append(str(row[word[1:-1]]))
            else:
                ret_prompt_words.append(word)
        final_prompt_list.append(' '.join(ret_prompt_words))
    
    return final_prompt_list

def answer_extraction_from_response_gpt_only(response_list, impute_col, qa_model = roberta_qa_model, qa_tokenizer = roberta_qa_tokenizer):
    extracted_answer_list = []
    for response in response_list:
        question = "What is the {} value for Tuple 1?".format(impute_col)
        context = response
        inputs = qa_tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0] # the list of all indices of words in question + context

        # text_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids) # Get the tokens for the question + context
        answer_start_scores, answer_end_scores = qa_model(**inputs, return_dict=False)

        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        if answer != "<s>":
            ret_response = answer
        else:
            ret_response = context

        extracted_answer_list.append(ret_response)

    return extracted_answer_list


def send_gpt_prompts_single(all_query_tuples_serialized, missing_att, custom_prompt_list = None):

    ### GPT3.5 Params
    service_name = os.getenv("SERVICE_NAME")
    deployment_name = os.getenv("DEPLOYMENT_NAME")
    key = os.getenv("API_KEY")  # please replace this with your key as a string or in .env file

    openai.api_key = key
    openai.api_base =  "https://{}.openai.azure.com/".format(service_name) # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    openai.api_type = 'azure'
    openai.api_version = '2022-06-01-preview' # this may change in the future

    deployment_id=deployment_name #This will correspond to the custom name you chose for your deployment when you deployed a model. 

    ### Main Loop
    final_answers = []

    if custom_prompt_list != None:
        for i in range(len(custom_prompt_list)):
            query = custom_prompt_list[i]
            prompt="<|im_start|>system\nThe system is an AI assistant that helps users get relevant information and answers.\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant".format(query)
            response1 = openai.Completion.create(engine="gpt3_davinci_imputer", prompt=prompt,
                                            temperature=0.2,
                                            max_tokens=32,
                                            top_p=0.95,
                                            frequency_penalty=0.5,
                                            presence_penalty=0.5,
                                            stop=["<|im_end|>"])
            txt = response1["choices"][0]["text"]
            final_answers.append(txt)
            time.sleep(0.3)
    
    else:
        for i in range(len(all_query_tuples_serialized)):
            query = all_query_tuples_serialized[i]
            prompt="<|im_start|>system\nThe system is an AI assistant that helps users get relevant information and answers.\n<|im_end|>\n<|im_start|>user\nTuple 1 = {} What should be the {} value for Tuple 1?\n<|im_end|>\n<|im_start|>assistant".format(query, missing_att)
            response1 = openai.Completion.create(engine="gpt3_davinci_imputer", prompt=prompt,
                                            temperature=0.2,
                                            max_tokens=32,
                                            top_p=0.95,
                                            frequency_penalty=0.5,
                                            presence_penalty=0.5,
                                            stop=["<|im_end|>"])
            txt = response1["choices"][0]["text"]
            final_answers.append(txt)
            time.sleep(0.3)

    extracted_final_repairs = answer_extraction_from_response_gpt_only(final_answers, missing_att)
    return extracted_final_repairs
        

print("LOADED gpt_only_helper")