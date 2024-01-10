from openai import OpenAI
import pandas as pd
import os 
import tqdm
import time
import re
import certifi
from pymongo import MongoClient, ReturnDocument
from ast import literal_eval
from dotenv import load_dotenv
from stopit import threading_timeoutable as timeoutable
from tenacity import (
    retry,
    wait_random_exponential,
)  # for exponential backoff

import os



# take environment variables from .env.
load_dotenv()  

clientAI = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
  organization = os.getenv("OPENAI_ORGANIZATION")
)

# openai.organization = os.getenv("OPENAI_ORGANIZATION")
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect database
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
database = os.getenv("MONGO_DATABASE")

client = MongoClient(
    f"mongodb+srv://{username}:{password}@talkcluster.els0bio.mongodb.net/",
)

db = client[database]


#get_explanationResponse
def get_explanationResponse(id, previous_exp, question_list, message_exp, history):
    
    explanationResponse, recordMessageExp, recordHisotry = explainerGPT(id, previous_exp, question_list, message_exp , history)

    return explanationResponse,recordMessageExp,recordHisotry

#get_summaryExplanationResponse
def get_summaryExplanationResponse(id, history, question_list):

    summaryExplanationResponse, recordHisotry = Summary_GPT(id, history, question_list)

    return summaryExplanationResponse, recordHisotry

def initialmessage(id):
    sample = db['questionCollectionData'].find_one({"questionCollection_id": id})
    claim = sample['claim']
    label = sample['label']
    evidences = sample['evidences']
    original_explanation = sample['explanations']

    prompt = f"""
As a fake news debunker, it is important to provide credible explanations.
This is the claim and label:
Claim: {claim}
Label: {label}
This is the list of evidence that supports or contradicts the claim:
--- Evidence List ---
{evidences}
--- Evidence List ---
Can you explain why this claim is this label with the evidence in one sentence? Make sure your explanation is included in the evidences list. Try to make it clear and short.
"""
    # print(prompt)
    message_exp=[]

    history = "\n**Conversation History**:\n"
    history += "explainer: [" + original_explanation +"]\n"
    message_ques = []
    return original_explanation, message_exp, message_ques, history


@retry(wait=wait_random_exponential(min=5, max=15))
def explainerGPT(id, previous_exp, question_list, message_exp , history):

    sample = db['questionCollectionData'].find_one({"questionCollection_id": id})
    claim = sample['claim']
    label = sample['label']
    evidences = sample['evidences']


    questionstr = ""
    questionstr += question_list
    
    history += f'Questioner: ["{questionstr}"]' + '\n'

    print("claim:" + claim)
    print("label:" + label)

    if isinstance(evidences, list):
        for index, evidence in enumerate(evidences):
            print(f"Evidence {index}: {evidence}")
    else:
        print("No evidences found or evidences is not a list.")    

    print("previous_exp:" + previous_exp)
    print('Round_history'+ history)

    prompt = f"""
As a fake news debunker, it is important to provide credible explanations.
This is the claim and label:
Claim: {claim}
Label: {label}
This is the list of evidence that supports or contradicts the claim:
--- Evidence List ---
{evidences}
--- Evidence List ---
This is the conversation history from you and questioner:
--- Conversation History ---
{history}
--- Conversation History ---
This is the last round of explanation and questioner response:
--- Last Explanation ---
{previous_exp}
--- Questioner Response ---
{questionstr}
--- Last Explanation ---
​
Can you explain why this claim is this label with the evidence in one sentence? Please generate a more convincing explanation based on the question raised by the questioner. Make sure your explanation is included in the evidences list. Try to make it clear and short.
"""
    message_exp=[
        {
            "role": "user", 
            "content":  prompt
        }
    ]

    while True:
        @timeoutable()
        def catchtimeout(message_exp):
                response_exp = clientAI.chat.completions.create(
                model="gpt-4-0613",
                messages = message_exp)
                return response_exp.choices[0].message.content
        
        response_exp = catchtimeout(message_exp, timeout=60)

        if response_exp != None:
            # print("response_exp",response_exp)
            break
        else:
            print("timeout")

    history += "explainer: [" + response_exp + "]\n"

    time.sleep(3)
    return response_exp, message_exp, history



@retry(wait=wait_random_exponential(min=5, max=15))
def Questioner_GPT(id, round, explanation, message_ques, history):
    global sample
    claim = sample['Claim'][id]
    add1 = ""
    add2 = ""
    if round > 1:
        add1 = """If at least one rating is 5 and there are no ratings below 4, you can respond with \"I agree with this explanation.\""""
        add2 = """If at least one rating is 5 and there are no ratings below 4, your response should be "Persuasiveness: <your persuasiveness rating>, Logical Correctness: <your logical correctness rating>, Completeness: <your completeness rating>, Conciseness: <your conciseness rating>, Question: <\"I agree with this explanation.\">"."""

    prompt = f"""
Assume you are a general newsreader.
You just saw this claim and have no preliminary knowledge.
Your task is to rate the persuasiveness and logical correctness of the explanation and ask a question if in need (rating below 5 and two 4s).
The explanation should be clear, short, and persuasive.
---
Your task is to rate the persuasiveness, logical correctness, completeness and conciseness of the explanation based on the following criteria:
--- Rating Criteria ---
Persuasiveness Criteria:
1 - Definitely not persuasive
2 - Probably not persuasive
3 - Might or might not be persuasive
4 - Probably persuasive
5 - Definitely persuasive
​
Logical Correctness Criteria:
1 - Definitely logically incorrect
2 - Probably logically incorrect
3 - Might or might not be logically correct
4 - Probably logically correct
5 - Definitely logically correct
​
Completeness Criteria:          
1 - Definitely not complete
2 - Probably not complete
3 - Might or might not be complete
4 - Probably complete
5 - Definitely complete
​
Conciseness Criteria:
1 - Definitely not concise
2 - Probably not concise
3 - Might or might not be concise
4 - Probably concise
5 - Definitely concise
--- Rating Criteria ---
​
Claim
---
{claim}
---
Explanation
---
{explanation}
---
Please review the claim and explanation, and rate the persuasiveness, logical correctness, completeness and conciseness of the explanation accordingly.
Please assess this explanation using the same criteria you used to rate the previous explanation in this conversation.
​
Then, ask a question to provide feedback about why the explanation is not persuasive, logically incorrect, incomplete or inconcise.
Please ensure that your questions enhance the explanation's logical correctness, persuasiveness, completeness, and conciseness.
Make sure your question doesn't share similar meanings with previous questions.
​
{add1}
---
Your response should be in the format: "Persuasiveness: <your persuasiveness rating>, Logical Correctness: <your logical correctness rating>, Completeness: <your completeness rating>, Conciseness: <your conciseness rating>, Question: <your Question>". 
{add2}
---
Response:
"""
    # print(prompt)
    message_ques.append(
        {
            "role": "user",
            "content":  prompt
        }
    )
    while True:
        @timeoutable()
        def call_GPT(message_ques):
                response_ques = clientAI.chat.completions.create(
                model="gpt-4-0613",
                messages = message_ques,
                temperature = 1)
                return response_ques.choices[0].message.content
        
        response_ques_content = call_GPT(message_ques, timeout=60)
        if response_ques_content != None:
            break
        else:
            print("timeout")
    message_ques.append(
        {
            "role": "assistant",
            "content":  response_ques_content
        }
    )
    # 提取 Persuasiveness 的评分值
    start_index = response_ques_content.find("Persuasiveness:") + len("Persuasiveness:")
    end_index = response_ques_content.find(",", start_index)
    persuasiveness_rating = int(response_ques_content[start_index:end_index].strip())

    # 提取 Logical Correctness 的评分值
    start_index = response_ques_content.find("Logical Correctness:") + len("Logical Correctness:")
    end_index = response_ques_content.find(",", start_index)
    logical_correctness_rating = int(response_ques_content[start_index:end_index].strip())

    # 提取 Completeness 的评分值
    start_index = response_ques_content.find("Completeness:") + len("Completeness:")
    end_index = response_ques_content.find(",", start_index)
    completeness_rating = int(response_ques_content[start_index:end_index].strip())

    # 提取 Conciseness 的评分值
    start_index = response_ques_content.find("Conciseness:") + len("Conciseness:")
    end_index = response_ques_content.find(",", start_index)
    conciseness_rating = int(response_ques_content[start_index:end_index].strip())

    rating = [persuasiveness_rating, logical_correctness_rating, completeness_rating, conciseness_rating]

    time.sleep(3)
    history += f"Questioner: [{response_ques_content}]" + '\n'
    return rating, response_ques_content, message_ques, history



@retry(wait=wait_random_exponential(min=5, max=15))
def Summary_GPT(id, history, question_list) :

    sample = db['questionCollectionData'].find_one({"questionCollection_id": id})
    claim = sample.get('claim', 'No claim found.')
    label = sample.get('label', 'No label found.')
    evidences = sample.get('evidences', 'No evidences found.')

    questionstr = ""
    questionstr += question_list
    history += f'Questioner: ["{questionstr}"]' + '\n'

    print("claim:" + claim)
    print("label:" + label)

    if isinstance(evidences, list):
        for index, evidence in enumerate(evidences):
            print(f"Evidence {index}: {evidence}")
    else:
        print("No evidences found or evidences is not a list.")    

    print('Round_history'+ history)

    prompt = f"""
As a fake news debunker, you need to generate the best explanation based on the conversation history of the questioner and explainer.
​
The explainer has generated an explanation based on the list of evidence and refined the explanation based on the questioner response.
The questioner has raised the question to seek clarification or additional information until the explanation is convincing.
​
This is the claim and label:
Claim: {claim}
Label: {label}
This is the list of evidence that supports or contradicts the claim:
--- Evidence List ---
{evidences}
--- Evidence List ---
This is the conversation history from explainer and questioner:
--- Conversation History ---
{history}
--- Conversation History ---
​
Can you explain why this claim is this label with the evidence in one sentence? Please summarize the conversation history and generate the best explanation. Make sure your explanation is included in the evidences list. Try to make it clear and short.
"""
    # print(prompt)
    message_summ=[
        {
            "role": "user", 
            "content":  prompt
        }
    ]
    while True:
        @timeoutable()
        def call_GPT(message_summ):
                final_explanation = clientAI.chat.completions.create(
                model="gpt-4-0613",
                messages = message_summ)
                return final_explanation.choices[0].message.content
        
        final_explanation = call_GPT(message_summ, timeout = 60)
        if final_explanation!= None:
            break
        else:
            print("timeout")
    final_explanation_content = final_explanation
    # final_explanation_content= final_explanation['choices'][0]['message']['content']

    history += "explainer: [" + final_explanation_content + "]\n"

    print("test_end:" + history)

    time.sleep(3)

    return final_explanation_content, history

