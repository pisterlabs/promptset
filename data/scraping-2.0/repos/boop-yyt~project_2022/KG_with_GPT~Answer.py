from binhex import openrsrc
from curses import init_pair
import os
from unittest import result
import openai
import json
import argparse
import Levenshtein
from time import sleep
import sacrebleu
# from nltk.translate.bleu_score import sentence_bleu

openai.api_key = os.getenv("OPENAI_API_KEY")
example ={
        "puzzle": ["A man walks into a bar and asks the bartender for a drink of water.",
                    "The bartender pulls out a qun, points it at the man and cocks it. ",
                    "The man pauses before saying \"Thank you\"and leaving.",
                    "What happened?"
        ],
        "final_answer": ["The man had the hiccups."
                  "The bartender realized this and chose instead to cure the hiccups by frightening the man with the gun."
        ],
        "question_list":["Could the bartender hear him?",
                       "Did the man ask for water in an offensive way?"],
        "answer_list":["Yes.","No."],
        "hint":"As we said, there is nothing wrong with either the building or the elevator. There is, however, some feature of the man that causes him to take the stairs from the seventh floor."
    }

def waitforGPT(sec):
    sleep(sec)

def load_data(data_path):
    with open(data_path, 'r') as f:
        data_set = json.load(f)
    return data_set

def extract_input_item(data):
    puzzle = "".join(data["puzzle"])
    truth = "".join(data["final_answer"])
    fol_q = list(data["question_list"])
    fol_a = list(data["answer_list"])
    return puzzle, truth, fol_q, fol_a

def get_response(prompt):
    output_ai = ""
    response_num = 0
    while output_ai == "" and response_num <= 3:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            logprobs = 1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n\n"]
            )
        output_ai = response.get('choices')[0]['text'].split("\n")[0]
        log_info = response.get('choices')[0]
        if output_ai == "":
            response_num += 1
            waitforGPT(10)
    return output_ai,log_info

def question_generation_input(data_item):
    task_description = "I am an intelligent bot that can play situation puzzles with user. A puzzle is given first, and the user begin to ask a \"yes/no\" question to ensure details."
    # task_description = "question generation"
    puzzle, truth, fol_q, fol_a = extract_input_item(data_item)
    example_prefix = "puzzle:" + "".join(example["puzzle"]) + "\n"
    example_prefix += "follow_up_Q:"+example["question_list"][0]+"\n"+"follow_up_A:"+example["answer_list"][0] + "\n"
    example_prefix += "follow_up_Q:"+example["question_list"][1] + "\n"
    input_prefix = "puzzle:" + puzzle + "\n"  
    if len(fol_q) and len(fol_a):
        for fq,fa in zip(fol_q, fol_a):
            input_prefix += "follow_up_Q:"+ fq
            input_prefix += "follow_up_A:"+ fa
        input_prefix += "\n"
    input_prefix += "follow_up_Q:"
    final_prefix = task_description + '\n' + example_prefix + input_prefix
    # print("---------- question generation ----------")
    # print(final_prefix)
    return final_prefix

def answer_generation_input(data_item):
    task_description = "I am an intelligent bot that can play as judge in situation puzzles with user. A puzzle is given first, and the user begin to ask a \"yes/no\" question to ensure details, and I will give \"Yes/No/Irrelevent\" as answer to questions."
    # task_description = "answer generation"
    puzzle, truth, fol_q, fol_a = extract_input_item(data_item)
    example_prefix = "truth:" + "".join(example["final_answer"]) + "\n"
    example_prefix += "follow_up_Q:"+example["question_list"][0]+ "\n" +"follow_up_A:"+example["answer_list"][0] + "\n"
    example_prefix += "follow_up_Q:"+example["question_list"][1]+ "\n" +"follow_up_A:"+example["answer_list"][1] + "\n"
    input_prefix = "truth:" + truth + "\n"  
    if len(fol_q) and len(fol_a):
        for fq,fa in zip(fol_q, fol_a):
            input_prefix += "follow_ip_Q:"+ fq
            input_prefix += "follow_ip_A:"+ fa
    input_prefix += "follow_up_Q:" + data_item["question_list"][-1] 
    input_prefix += "\n" + "follow_up_A:"
    final_prefix = task_description + '\n' + example_prefix + input_prefix
    # print("---------- answer generation ----------")
    # print(final_prefix)
    return final_prefix
    
def solution_generate_input(data_item, shuffle):
    task_description = "I am an intelligent bot that can play situation puzzles with user. A puzzle is given first, and the user begin to ask \"yes/no\" question to ensure details, then I will give the question a\"yes/no/irrelevent\" answer. Finally user try to give solution for the puzzle."
    puzzle, truth, fol_q, fol_a = extract_input_item(data_item)
    if shuffle == "True":
        randomnum = random.randint(0,5)
        random.seed(randomnum)
        random.shuffle(fol_q)
        random.seed(randomnum)
        random.shuffle(fol_a)
    example_prefix = "puzzle:" + "".join(example["puzzle"]) + "\n"
    example_prefix += "follow_up_Q:"+example["question_list"][0]+"\n"+"follow_up_A:"+example["answer_list"][0] + "\n"
    example_prefix += "solution:" + "".join(example["final_answer"]) + "\n"
    input_prefix = "puzzle:" + puzzle + "\n"  
    if len(fol_q) and len(fol_a):
        for fq,fa in zip(fol_q, fol_a):
            input_prefix += "follow_up_Q:"+ fq +"\n"
            input_prefix += "follow_up_A:"+ fa+"\n"
        # input_prefix += "\n"
    input_prefix += "solution:"
    final_prefix = task_description + '\n' + example_prefix + input_prefix
    # print("---------- solution generation ----------")
    # print(final_prefix)
    return final_prefix

