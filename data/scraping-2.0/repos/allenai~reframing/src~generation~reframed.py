import os
import json
import argparse
import sys
import torch
sys.path.insert(1, "src/encoding")
from encodeinstruction_reframed import encodeinstruction
import openai
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer
import time


list_task = ['subtask002_quoref_answer_generation', 'subtask003_mctaco_question_generation_event_duration', 'subtask005_mctaco_wrong_answer_generation_event_duration', 'subtask008_mctaco_wrong_answer_generation_transient_stationary', 'subtask022_cosmosqa_passage_inappropriate_binary', 'subtask033_winogrande_answer_generation', 'subtask034_winogrande_question_modification_object', 'subtask039_qasc_find_overlapping_words', 'subtask040_qasc_question_generation', 'subtask044_essential_terms_identifying_essential_words', 'subtask045_miscellaneous_sentence_paraphrasing', 'subtask052_multirc_identify_bad_question']


global generator
global tokenizer

def load_model(model_name):
    global generator
    global tokenizer
    device = 0 if torch.cuda.is_available() else -1
    print(device)
    generator = pipeline('text-generation', model=model_name, device = device)
    #set_seed(42)

def get_responses_gpt2(args, instruction, task):
    try:
        length_instruction = len(generator.tokenizer(instruction)['input_ids'])
        #print(instruction)
        max_token = 16
        if task == 'subtask022_cosmosqa_passage_inappropriate_binary' or task == 'subtask005_mctaco_wrong_answer_generation_event_duration' or task == 'subtask008_mctaco_wrong_answer_generation_transient_stationary' or task == 'subtask033_winogrande_answer_generation' or task == 'subtask039_qasc_find_overlapping_words' or task == 'subtask052_multirc_identify_bad_question':
            max_token = 3
        elif task == 'subtask044_essential_terms_identifying_essential_words' or task == 'subtask002_quoref_answer_generation':
            max_token = 10
        else:
            max_token = 30
        output = generator(instruction, truncatation = True, return_full_text = False, max_length = length_instruction + max_token)
        return output[0]['generated_text']
    except:
        return "error occured"

def get_responses_gpt3(args, instruction, task, engine):
    openai.api_key = args.API_TOKEN
    max_token = 16
    if task == 'subtask022_cosmosqa_passage_inappropriate_binary' or task == 'subtask005_mctaco_wrong_answer_generation_event_duration' or task == 'subtask008_mctaco_wrong_answer_generation_transient_stationary' or task == 'subtask033_winogrande_answer_generation' or task == 'subtask039_qasc_find_overlapping_words' or task == 'subtask052_multirc_identify_bad_question':
        max_token = 3
    elif task == 'subtask044_essential_terms_identifying_essential_words' or task == 'subtask002_quoref_answer_generation':
        max_token = 10
    else:
        max_token = 30
    try:
        time.sleep(2)
        response = openai.Completion.create(
                                            engine=engine,
                                            prompt=instruction,
                                            temperature=0.7,
                                            max_tokens=max_token,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0,
                                            stop = "\n"
    )
        choices = response.get('choices',0)
        if choices != 0:
            answer = choices[0]['text'].strip()
        else:
            answer = choices
            
        return answer
    except Exception:
        return "error occured"       
        
def get_answer(args, instruction, task):
    
    if  args.model_name.lower() == "gpt3_davinci":
        answer = get_responses_gpt3(args, instruction, task, engine = "davinci")
    
    if  args.model_name.lower() == "gpt3":
        answer = get_responses_gpt3(args, instruction, task, engine = "text-davinci-001")
    
    if "gpt2" in args.model_name.lower():
        answer = get_responses_gpt2(args, instruction, task)
    
    return answer


def generate_responses(args):
    isExist = os.path.exists("output_files_reframed")
    
    if not isExist:
        os.makedirs("output_files_reframed")
        os.makedirs("output_files_reframed/" + args.model_name)
    
    isExist = os.path.exists("output_files_reframed/"+ args.model_name)
    if not isExist:
        os.makedirs("output_files_reframed/" + args.model_name)
        
    
    q = 0
    start = int(args.start)
    end = int(args.end)
    
    for task in list_task[start:end]:
        
        task_answers = []
        print(task)
        
        if task == "subtask002_quoref_answer_generation" and "gpt2" in args.model_name:
            task_instructions = encodeinstruction(task, model_name = args.model_name, number_of_examples = 0, number_of_instances = int(args.number_of_instances))
        elif task == "subtask052_multirc_identify_bad_question" and "gpt2" in args.model_name:
            task_instructions = encodeinstruction(task, model_name = args.model_name, number_of_examples = 2, number_of_instances = int(args.number_of_instances))
        else:
            task_instructions = encodeinstruction(task, model_name = args.model_name, number_of_examples = int(args.number_of_examples), number_of_instances = int(args.number_of_instances))
        
        p = 0
        
        print(task_instructions[0])
        print("\n\n")
        for instruction in task_instructions:
            
            if q == 0 and "gpt2" in args.model_name:
                print("loading model")
                load_model(args.model_name)
                q = 1
                
            answer = get_answer(args, instruction, task)
            p = p + 1
            
            task_answers.append(answer)
        
        with open('output_files_reframed/' + args.model_name + "/" +task+'_prediction.json', 'r', encoding='utf-8') as f:
                true_answers = json.load(f)["true"]
                pred_length = len(task_answers)
                true_length = len(true_answers)
                
                if pred_length == true_length:
                    print("EQUAL LENGTH")
                else:
                    print("UNEQUAL LENGTH")
        with open('output_files_reframed/' + args.model_name + "/" +task+'_prediction.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps({"true": true_answers, "prediction": task_answers}, ensure_ascii=False, indent = 4))
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='baseline_gpt3')
    
    #parser.add_argument('--generate',help='1 if you want to generate answers')
    parser.add_argument('--API_TOKEN',help='API token for the model to be used')
    parser.add_argument('--model_name',help='API token for the model to be used')
    parser.add_argument('--number_of_examples',help='API token for the model to be used')
    parser.add_argument('--number_of_instances',help='API token for the model to be used')
    parser.add_argument('--start',help='API token for the model to be used', default = 0)
    parser.add_argument('--end',help='API token for the model to be used', default = 12)
    args = parser.parse_args()
    generate_responses(args)