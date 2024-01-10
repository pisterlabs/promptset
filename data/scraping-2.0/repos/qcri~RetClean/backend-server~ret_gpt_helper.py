import torch
import openai
import time
from dotenv import load_dotenv
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from utils import *
from retrieval_helper import * 
from reranker_helper import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_dotenv()

# Load Post Processing Models
roberta_qa_model = AutoModelForQuestionAnswering.from_pretrained("shamz15531/roberta_repair_extractor")
roberta_qa_tokenizer = AutoTokenizer.from_pretrained("shamz15531/roberta_repair_extractor")


# Helper Func to filter out not matched responses from GPT
def keep_first_positive_response(reponses_list):
    final_ret_responses = []
    for i in range(len(reponses_list)):
        query_responses = reponses_list[i]
        if len(query_responses) == 0:
            final_ret_responses.append("No Valid Response")
        else:
            added = False
            for j in range(len(query_responses)):
                response = query_responses[j]["repair"].replace("\n","").lower().strip()
                if len(response) > 5 and response[0:3] == "yes" and added == False: # heurisitc based filtering
                    to_add = query_responses[j].copy()
                    to_add["repair"] = to_add["repair"].replace("\n","")
                    final_ret_responses.append(to_add)
                    added = True
    return final_ret_responses

# Helper Func for post processing (answer extraction from GPT response)
### Function: Extracts relevant answer from a longer response
def answer_extraction_from_response(response_list, impute_col, qa_model = roberta_qa_model, qa_tokenizer = roberta_qa_tokenizer):
    extracted_answer_list = []
    for response in response_list:
        question = "What is the {} value for Tuple 1?".format(impute_col)
        context = response["repair"] 
        inputs = qa_tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0] # the list of all indices of words in question + context

        # text_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids) # Get the tokens for the question + context
        answer_start_scores, answer_end_scores = qa_model(**inputs, return_dict=False)

        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        ret_response = response.copy()
 
        if answer != "<s>":
            ret_response["repair"] = answer

        extracted_answer_list.append(ret_response)

    return extracted_answer_list


def send_gpt_prompts_with_ret(all_query_tuples_serialized, the_encoder, the_tokenizer, missing_att,
                              reranker_type = None, index_name = "es_index_1", index_type = "ES", object_imp = "object"):
    ### GPT3.5 Params
    service_name = os.getenv("SERVICE_NAME")
    deployment_name = os.getenv("DEPLOYMENT_NAME")
    key = os.getenv("API_KEY")  # please replace this with your key as a string or in .env file

    openai.api_key = key
    openai.api_base =  "https://{}.openai.azure.com/".format(service_name) # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    openai.api_type = 'azure'
    openai.api_version = '2022-06-01-preview' # this may change in the future

    deployment_id=deployment_name #This will correspond to the custom name you chose for your deployment when you deployed a model. 
    
    if reranker_type != None:
        reranker_model = load_encoder_reranker(mode=reranker_type)

    # 2D list for results of each query tuple
    all_final_results = []
    for i in range(len(all_query_tuples_serialized)):
        temp_results = []

        query_tuple = all_query_tuples_serialized[i]

        retrieved = search_index(query_tuple, # str format, serialized
                         "./tmp/aggregation_{}.csv".format(index_name),
                         "./faiss_index/", 
                         index_name, #es_attempt_2 for es ,  faiss_attempt_4 for faiss
                         the_encoder,
                         the_tokenizer,
                         index_type = index_type,
                         k = 3 # number of tuples to retrieve
                         )

        if reranker_type == "colbert":
            retrieved = colbert_like_rerank(query_tuple, retrieved, reranker_model)
        elif reranker_type == "crossencoder":
            retrieved = cross_encoder_based_rerank(query_tuple, retrieved, reranker_model)

        for j in range(3):
            ret = retrieved["serialization"][j]
            prompt="<|im_start|>system\nThe system is an AI assistant answer questions based on the information in the 2 Tuples provided. Only answer Question 2 if you answer yes to Question 1. If your answer to Question 1 in 'no', response with that.\n<|im_end|>\n<|im_start|>user\nTuple 1 = {} Tuple 2 = {}. Question 1: Are Tuple 1 and Tuple 2 about the same {}, yes or no? Question 2: If your answer to Question 1 was yes, then determine what the {} value for Tuple 1 should be based on Tuple 2? \n<|im_end|>\n<|im_start|>assistant".format(query_tuple, ret, object_imp, missing_att)
            ## Send GPT3.5 Request
            response1 = openai.Completion.create(engine="gpt3_davinci_imputer", prompt=prompt,
                                        temperature=0.1,
                                        max_tokens=32,
                                        top_p=0.95,
                                        frequency_penalty=0.5,
                                        presence_penalty=0.5,
                                        stop=["<|im_end|>"])

            temp_results.append({
               "repair" : response1["choices"][0]["text"],
               "source" : ret,
               "table" : retrieved["table"][j],
               "index" : retrieved["index"][j]
            })
            time.sleep(0.5)
        all_final_results.append(temp_results)
    
    # Keep First Positive Responses Only for Each Query Tuple
    all_final_results_positive_only = keep_first_positive_response(all_final_results)

    # Extract Answers
    all_final_results_positive_only_post_processed = answer_extraction_from_response(all_final_results_positive_only,missing_att)
    

    return all_final_results_positive_only_post_processed

print("LOADED ret_gpt_helper")