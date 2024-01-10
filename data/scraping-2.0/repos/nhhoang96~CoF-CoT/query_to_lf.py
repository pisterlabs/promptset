import json
import os
import time
import argparse
import re
from collections import Counter
import numpy as np
import copy
import pandas as pd

import openai
import pprint                  
import google.generativeai as palm

structure_map={'amr': 'Abstract Meaning Representation (AMR) Graph in the textual Neo-Davidson format', 'dp': 'Dependency Parsing Graph', 'cp': 'Constituency Parsing Graph', 'none':''}

def get_intent_slot_vob(dataset):
    if dataset == "MTOP":
        intent_vocab, slot_vocab = [], []
        intent_file = './nlu_data/mtop_flat_simple/intent_vocab.txt'
        slot_file = './nlu_data/mtop_flat_simple/slot_vocab.txt'

        for line in open(intent_file, 'r'):
            intent_vocab.append(line.strip())

        for line in open(slot_file, 'r'):
            slot_vocab.append(line.strip())

        intent_map_file = './nlu_data/mtop_flat_simple/intent_vocab_map.jsonl'
        intent_vocab, intent_descr=[],[]
        intent_map={}
        intent_rev_map={}
        for line in open(intent_map_file, 'r'):
            line_output = json.loads(line)
            name, description = line_output['label'].strip(), line_output['label_description'].strip()
            intent_map[name] = description
            intent_vocab.append(name)
            intent_descr.append(description)
            intent_rev_map[description] = name

        

        slot_map={}
        slot_map_file = './nlu_data/mtop_flat_simple/slot_vocab_map.jsonl'

        slot_vocab, slot_descr=[],[]
        slot_rev_map={}
        for line in open(slot_map_file, 'r'):
            line_output = json.loads(line)
            name, description = line_output['label'].strip(), line_output['label_description'].strip()
            slot_map[name] = description
            slot_vocab.append(name)
            slot_descr.append(description)
            slot_rev_map[description] = name

        print ("Slot vocab", slot_vocab)

    elif dataset == "MASSIVE":
        # ----- MASSIVE -----#

        intent_vocab, slot_vocab = [], []
        intent_file = './nlu_data/massive_data_full/intent_vocab.txt'
        slot_file = './nlu_data/massive_data_full/slot_vocab.txt'

        for line in open(intent_file, 'r'):
            intent_vocab.append(line.strip())

        for line in open(slot_file, 'r'):
            slot_vocab.append(line.strip())


        intent_map_file = './nlu_data/massive_data_full/intent_vocab_map.jsonl'
        intent_vocab, intent_descr=[],[]
        intent_map={}
        intent_rev_map={}
        for line in open(intent_map_file, 'r'):
            line_output = json.loads(line)
            name, description = line_output['label'], line_output['label_description']
            intent_map[name] = description
            intent_vocab.append(name)
            intent_descr.append(description)
            intent_rev_map[description] = name
        

        slot_map={}
        slot_map_file = './nlu_data/massive_data_full/slot_vocab_map.jsonl'

        slot_vocab, slot_descr=[],[]
        slot_rev_map={}
        for line in open(slot_map_file, 'r'):
            line_output = json.loads(line)
            name, description = line_output['label'], line_output['label_description']
            slot_map[name] = description
            slot_vocab.append(name)
            slot_descr.append(description)
            slot_rev_map[description] = name

    return intent_vocab, intent_descr, intent_map, intent_rev_map, slot_vocab, slot_descr, slot_map, slot_rev_map


def call_openai(args, que_promp, output_num, temperature):
    #print ("OPENAI call")
    success = False
    while success == False:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": que_promp}
                ],
                n = output_num,
                temperature = temperature,
            )
            success = True
        except:
            time.sleep(1)
    if (output_num == 1):
        return response['choices'][0]['message']['content']
    else: #TODO: Multiple outputs for future consistency/ majority voting performance

        predictions = get_generated_list(response['choices'])
        output = find_majority(predictions)
        
        return output

def call_palm(args,model, context, sent_input, full_prompt=None):
    print ("Input prompt",  context + sent_input)
    completion = palm.chat(
                    model=model,
                    context=context,
                    messages=sent_input,
                    temperature=0.7,
                    # The maximum length of the response
                    #max_output_tokens=800,
                    candidate_count=1
                )


    output = completion.last
    if (output == None):
        output=''
    #print ("Completion check", completion)
    return output


def get_generated_list(response_list):
    que_list = []
    for que in response_list:
        que_list.append(que['message']['content'])
    return que_list


def find_majority(inputs):
    counter = Counter(inputs)
    majority = counter.most_common(1)
    return majority[0][0]


def parse_lf(lf):
    #print ("LF", lf)
    intent_slot = re.sub(r'[\[\]]',' ',lf).strip()
    intent_slot = re.sub('\s\s+', ' ', intent_slot)
    intent_slot = intent_slot.split(' ')
    slots = []
    slot_pairs=[]
    intent=''
    slot_val=''
    slot_vals=[]
    for item in intent_slot:
        item = item.replace('[','')
        item = item.replace(']','')
        if (item.startswith('IN:')):
            intent = item.split(':')[-1].lower()
        elif (item.startswith('SL:')): # slot type
            if (len(slot_pairs) >= 1):
                slot_pairs.append(slot_val.strip())
                slot_vals.append(slot_val)
                slots.append(tuple(slot_pairs))  
                slot_pairs= []
            slot_pairs.append(item.split(':')[-1].lower())

            #reset
            slot_val = ''
        else:
            slot_val+= ' ' + item

    # Final slots
    slot_pairs.append(slot_val.strip())
    slot_vals.append(slot_val.strip())
    slots.append(tuple(slot_pairs))
    return intent, slots, slot_vals

def condition_intent_info(args, current_prompt,intent_info, intent_map):
    output_prompt = current_prompt
    output_prompt += 'Potential Intent Types: ' + intent_info + '\n'
    return output_prompt


def reformat_output(output):
    core_output = output.split('\n\n')[1]
    print ("Core output", core_output)
    clear_str = core_output.replace('[','').replace(']','')
    clear_str = clear_str.replace('`', '`')                                                                                                                                                                                                                                                                                                                      
    slots=[]             
    intent=''            
    for element in clear_str.split('\n'):
        if (element.startswith('IN:')):
            intent = element.split(':')[-1].strip()
        else:            
            slot_info = element.split(':')[-1].strip()
            slot_info = slot_info.replace('"','')
            slot_info = slot_info.replace('=', '')
            slot_info = re.sub(' +', ' ', slot_info)
            slot_info = slot_info.strip()
            slots.append(slot_info)
                         
    return_output = '[IN:' + str(intent)
    for s in slots:      
        return_output += ' [SL:' + s + ']'
    return_output += ' ]'
                         
    print ("Return output", return_output)
    return return_output


def load_model(args):
    if (args.model_type == 'openai'):
        key_file = open('./openai_k.txt', 'r')
        key = [k.strip() for k in key_file][0]
        openai.api_key =  str(key)
        model_name = "gpt-3.5-turbo"
        model = "gpt-3.5-turbo"
    else:
        key_file = open('./google_k.txt', 'r')
        key = [k.strip() for k in key_file][0]
        palm.configure(api_key = key)                                 
        model_name = 'models/chat-bison-001'
        model = palm.get_model(model_name)
        print(model)                   
    return model



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="MTOP", choices=["MTOP", "MASSIVE"], type=str,
                    help='Type of dataset')

parser.add_argument("--demo_file", default='demo_5_label', type=str, help='Kind of conditioning: (none, control, control_filter)')

parser.add_argument("--type_prompt", default='chain_of', type=str, help='Direct Prompting or CoT')
parser.add_argument('--seed', default=111, type=int, choices=[111,222,333])

parser.add_argument("--test_file", default='test', type=str, help='Kind of conditioning: (none, control, control_filter)')
parser.add_argument("--type_condition", default='control_single', type=str, help='Kind of conditioning: (none, control, control_filter)')
parser.add_argument("--add_demo", choices=['true','false'], default='false', type=str)
parser.add_argument("--output_for", choices=['api','test'], default='test', type=str)

parser.add_argument("--voting_method", default='major', type=str)
parser.add_argument("--structure_rep",  choices=['amr','dp','cp','none'], default='amr', type=str)
parser.add_argument("--number_output",  default=10, type=int)
parser.add_argument("--number_demo",  default=11, type=int)
parser.add_argument('--demo_select_criteria', default='random', type=str)
parser.add_argument("--temperature",  default=0.7, type=float)

parser.add_argument('--add_domain', default='true', type=str)

parser.add_argument('--write_output', default='false', type=str)

parser.add_argument('--model_type', default='openai', type=str)

args = parser.parse_args()

#------ Prepare Demos -----#
if (args.dataset == 'MTOP'):
    data_root = './nlu_data/mtop_flat_simple/'
else:
    data_root = './nlu_data/massive_data_full/'

input_fs_file = os.path.join(data_root, args.demo_file + '.txt')

# ----- Output file definition ------#
input_test_file = os.path.join(data_root, args.test_file + '_'  + str (args.seed) + '.txt')
print ("Evaluation File:", input_test_file)
selected_demo_dict_ex={'utt':[], 'intent':[], 'key_phrase':[], 'pair':[], 'AMR':[], 'lf':[], 'utt_length':[], 'domain':[]}
ex_counter=0
for line in open(input_fs_file, 'r'):
    if (args.dataset == 'MTOP'):
        utt,lf,_,domain,_,amr_info = line.strip().split('\t')
    else:
        utt,lf,_,domain,amr_info = line.strip().split('\t')

    print ("ex", utt, lf, domain, amr_info)
    intent, slot_pairs, slot_vals = parse_lf(lf)
    

    selected_demo_dict_ex['utt'].append(utt)
    selected_demo_dict_ex['intent'].append(intent)
    selected_demo_dict_ex['key_phrase'].append(','.join(slot_vals))
    selected_demo_dict_ex['pair'].append(','.join(map(str, slot_pairs)))
    selected_demo_dict_ex['AMR'].append(amr_info)
    selected_demo_dict_ex['lf'].append(lf)

    selected_demo_dict_ex['domain'].append(domain)
    selected_demo_dict_ex['utt_length'].append(len(utt.split(' ')))

    ex_counter += 1

if (args.add_demo == 'true'):
    print ("Selected key-len %d \t Num samples:%d"%(len(selected_demo_dict_ex), len(selected_demo_dict_ex['utt'])))
    print ("Ex counter", ex_counter)

#
## OpenAPI api 
if (args.output_for == 'api'):
    model = load_model(args)

# ---- Generic Structure
intent_vocab, intent_descr, intent_map, intent_rev_map, slot_vocab,slot_map, slot_descr,slot_rev_map = get_intent_slot_vob(args.dataset)
intent_str = ','.join(intent_vocab)
intent_descr_str=','.join(intent_descr)
slot_str = ','.join(slot_vocab)


#--- Instruction Prompt


#------ STEP 1: AMR REP INSTRUCTIONS ------------ #
gen_step_1bc = 'Given the sentence and its potential intent types, generate one and only one ' + structure_map[args.structure_rep]

if (args.structure_rep == 'amr'):
    gen_step_1bc += '. The format includes :ARG and :op. Each word in the leaf node must exist in the given sentence '
    gen_step_1bc +='Return in the following format.\nAMR Graph: ```amr_graph```'

gen_step_1bc += '. No explanation is needed. Return 1 single AMR. \n'

#----- STEP 2: INTENT GENERAL INSTRUCTIONS ------- #
gen_step_1c = 'Given the intent vocabulary '
if (args.add_domain == 'true'):
    gen_step_1c += ', domain name, '

gen_step_1c += 'and the input sentence, choose 1 of the following in the intent vocabulary as the intent name for the sentence. Return one single exact match intent type existent in the intent vocabulary. Do not provide explanation. Do not return any additional information that is not asked. Return shorter than 1 sentence. Do not mention logic form. Return in the format: Intent: ``intent_name`` \n'
gen_step_1c += '\n'
gen_step_1c += 'Intent Vocabulary: ' + intent_str + '\n'


#------ STEP 3: GENERATE KEY PHRASES ------------ #
gen_step_2 = 'Based on the sentence and its ' + structure_map[args.structure_rep]

gen_step_2_nostruct = 'Based on the sentence, identify a list of key phrases for the sentence. Each word in key phrases must exist in the given sentence and might only appear once in the returned list. Key phrases need to contain consecutive words in the given sentence. Return a list of key phrases separated by commas \n'

gen_step_2c = 'Based on the sentence, its potential intents and its ' + structure_map[args.structure_rep]

if (args.add_domain == 'true'):
    gen_step_2 += ',its domain name'
    gen_step_2c += ',its domain name'

gen_step_2 += ', identify a list of key '

gen_step_2c += ', identify a list of key '
if (args.structure_rep == 'amr'):

    gen_step_2+= 'noun '
    gen_step_2c += 'noun '


gen_step_2 += 'phrases for the sentence. Key phrases can be made up from multiple AMR concepts. Each word in key phrases must exist in the given sentence. Each word in the sentence appears in only one key phrase. Key phrases need to contain consecutive words in the given sentence. Key phrases do not need to cover all words in the sentence. Return a list of key phrases separated by commas. '
gen_step_2 += "Output cannot be longer than 1 sentence."
gen_step_2 +='\n'

gen_step_2c += 'phrases for the sentence. Key phrases can be made up from multiple AMR concepts. Each word in key phrases must exist in the given sentence. Each word in the sentence must appear in at most one and only one key phrase. Key phrases need to contain consecutive words in the given sentence. Key phrases do not need to cover all words in the sentence. Return a list of key phrases separated by commas.'
gen_step_2c += 'Output cannot be longer than 1 sentence.'
gen_step_2c += '\n'


#------ STEP 4: GENERATE LABELS FOR KEY PHRASES ------------ #
gen_step_3c = 'Given the slot vocabulary'
gen_step_3c+= ', the sentence, its potential intents, its ' + structure_map[args.structure_rep] +', its key phrases'
if (args.add_domain == 'true'):
    gen_step_3c += ', and domain name'

gen_step_3c += ', identify one of the following in the slot vocabulary as the slot type for each key phrase. Return the list of key phrases and their corresponding slot types in the following format: (slot_type: key_phrase) separated by commas. Chosen slot type must exist in the vocabulary.Slot type always stands before slot value.'


gen_step_3c += '\n'
gen_step_3c += 'Slot Vocabulary: ' + slot_str + '\n'


#------ STEP 5: GENERATE LABELS FOR KEY PHRASES ------------ #
gen_step_4 = 'Given the sentence, its potential intent types, its slot type and slot value pairs in (slot_type, slot_value) format,'
if (args.add_domain == 'true'):
    gen_step_4 += ' domain,'


gen_step_4 += 'generate a single logic form in the format: [IN:intent_name [SL:slot_type slot_value][SL:slot_type slot_value]] where IN: is followed by an intent type and SL: is followed by a slot type and slot value pair separated by white space. The number of [SL: ] is unlimited. The number of [IN: ] is limited to 1. Use the given information precisely.'
gen_step_4 +=' Return in the following format: \n Output Format: Logic Form: ```logic_form``` '

new_session="New Session: "
direct_prompt = "Given the intent type vocabulary, slot type vocabulary and sentence, generate logic form of the sentence in the format of [IN:intent_name [SL:slot_type slot_value] [SL:slot_type slot_value]] where IN: is followed by an intent type and SL: is followed by a slot type and slot value pair separated by white space. The number of [SL: ] is unlimited. The number of [IN: ] is limited to 1 \n"
direct_prompt += "Intent Type: " + intent_str + "\n"
direct_prompt += "Slot Type: " + slot_str + "\n"

type_prompt,seed_val = args.type_prompt, args.seed
add_demo, condition_type = args.add_demo, args.type_condition
add_voting, numdemo, select = args.voting_method, args.number_demo, args.demo_select_criteria
struc_rep = args.structure_rep
num_output =args.number_output
add_domain=args.add_domain
model_type = args.model_type

if (args.add_demo == 'false'):
    result_output_file = f'./result_{model_type}_{args.dataset}/zs_ours_noamr_{seed_val}.jsonl'
else:
    result_output_file = f'./result_{model_type}_{args.dataset}/fs_ours_noamr_{seed_val}.jsonl'

if (args.write_output == 'true'):
    writer = open(result_output_file, 'w')
else:
    writer=None
slot_type=''
intent=''
amr_graph=''
key_phrases=''



def gen_step_1a(args,intent_str):

    if (args.type_condition == 'none'):
        step_1a_prompt = gen_step_1 + 'Intent Vocabulary: ' + intent_str + '\n'

    elif (args.type_condition == 'control_single'):
        step_1a_prompt = gen_step_1 + 'Intent Vocabulary: ' + intent_str + '\n'
    elif (args.type_condition == 'control'):
        step_1a_prompt = gen_step_1c + 'Intent Vocabulary: ' + intent_str + '\n'
    elif (args.type_condition == 'control_filter'):
        step_1a_prompt = gen_step_1c_filter + 'Intent Vocabulary: ' + intent_str + '\n'
    #if (demo == 'false'):
    #step_1a_prompt += 'Sentence: ' + utterance + '\n'
    #step_1a_prompt += 'Intent type: ' + '\n'
    #prompt = 
    return step_1a_prompt




def gen_step_prompt(args, step_number='1a'):
    
    if (step_number == '1a'):
        out_prompt = gen_step_1c

    elif (step_number == '2'):
        if (args.type_condition == 'none'):
            out_prompt = gen_step_2
        else:
            out_prompt = gen_step_2c

    elif (step_number == '3'):
        out_prompt = gen_step_3c
        step_3_prompt = gen_step_3c + 'Sentence: ' + utterance + '\n'

    elif (step_number == '4'):
        out_prompt = gen_step_4
    out_prompt += '\n'
    return out_prompt


def extract_intent(input_text):
    input_text = input_text.strip()
    m = re.findall(r"\[([\s\w]*)\]", input_text)

    return intent_out.strip()

def reformat_output(output, slot_vocab):
    if ('\n\n' in output):
        output = output.split('\n\n')[1]
    
    #core_output = output.split('\n\n')[1]
    core_output = output
    core_output = core_output.replace('```', '')
    clear_str = core_output.replace('[','').replace(']','')
    clear_str = clear_str.replace('`', '`')

    slots=[]
    intent=''
    for element in clear_str.split('\n'):
        element = element.strip()
        if (element.startswith('IN:')):
            intent = element.split(':')[-1].strip()
        else:
            slot_info = ' '.join(element.split(':')[1:]).strip()
            slot_info = slot_info.replace('"','')
            slot_info = slot_info.replace('=', ' ')
            slot_info = slot_info.replace(':', ' ')
            slot_info = re.sub(' +', ' ', slot_info)
            slot_info = slot_info.strip()
            slots.append(slot_info)
 
    return_output = '[IN:' + str(intent)
    for s in slots:
        return_output += ' [SL:' + s + ']'
    return_output += ' ]'    
    return return_output


#------------- Individual Step -----------------#


def gen_intent_prompt(args, selected_demo_dict_ex, domain_name):

    # --- Demo Step 1a
    if (args.add_demo == 'true'):
        demo_1 = gen_step_1c
        iter_demo=0
        for idx in range (len(selected_demo_dict_ex['utt'])): 
            demo_utt = selected_demo_dict_ex['utt'][idx]
            demo_intent = selected_demo_dict_ex['intent'][idx]
            demo_domain = selected_demo_dict_ex['domain'][idx]

            demo_1 += 'Sentence: ' + demo_utt + '\n'
            demo_1 += 'Intent type: ' + demo_intent + '\n'
            demo_1 += 'Domain: ' + demo_domain + '\n##\n'
            iter_demo += 1
        step_1a_prompt = demo_1 + '\n'
    else:
        step_1a_prompt = gen_step_1c

    step_1a_prompt = new_session + step_1a_prompt

    ori_step_1a = step_1a_prompt
    step_1a_prompt += 'Sentence: ' + utterance + '\n'
     
    sent_input =  'Input Sentence: ' + utterance + '\n'
    if (args.add_domain == 'true'):
        step_1a_prompt += 'Domain Name: ' + domain_name + '\n'
        sent_input += 'Domain Name: ' + domain_name + '\n'

    print ("===========================================================================")
    step_1a_prompt += 'Intent type:' + '\n'
    sent_input += 'Intent:' + '\n'
    if (args.output_for == 'api'):
        if (args.model_type == 'openai'):
            intent = call_openai(args,step_1a_prompt, args.number_output, args.temperature)
            print("STEP 1a: Get Intent ")
            print (step_1a_prompt)
        else:
            intent = call_palm(args, model,ori_step_1a, sent_input, step_1a_prompt)
            intent = intent.split('\n')[0]
            intent = intent.split(":")[-1].strip()
        print ("-- OUTPUT INTENT", intent)
    else:
        print("STEP 1a: Get Intent \n", step_1a_prompt)
        intent =''

    return intent


def generate_amr(args, selected_demo_dict_ex, domain_name):
    sent_input = ''
    # --- Step 1b: Get AMR Graph (except for no structure conditioning)
    if (args.structure_rep != 'none'):
        #--- Demo 1b
        if (args.add_demo == 'true'):
            iter_demo = 0
            demo_1b = gen_step_1bc

            for idx in range (len(selected_demo_dict_ex['utt'])): 
                demo_utt = selected_demo_dict_ex['utt'][idx]
                demo_intent = selected_demo_dict_ex['intent'][idx]
                demo_amr = selected_demo_dict_ex['AMR'][idx]
                demo_domain = selected_demo_dict_ex['domain'][idx]
                demo_1b += 'Sentence: ' + demo_utt + '\n'
                if (args.type_condition != 'none'):
                    demo_1b = condition_intent_info(args, demo_1b,demo_intent, intent_map)

                demo_1b += 'Domain Name: ' + demo_domain + '\n'
                demo_1b += structure_map[args.structure_rep] + ': ' + demo_amr + '\n##\n'
                iter_demo += 1

            assert iter_demo == args.number_demo

            step_1b_prompt = demo_1b + '\n'

        else:
            step_1b_prompt = gen_step_1bc
        ori_step_1b= step_1b_prompt
        step_1b_prompt += 'Sentence: ' + utterance + '\n'
        sent_input = 'Sentence: ' + utterance + '\n'


        if (args.add_domain == 'true'):
            step_1b_prompt += 'Domain Name: ' + domain_name + '\n'
        
        print ("===========================================================================")
        sent_input += structure_map[args.structure_rep] + ': ' + '\n'

        if (args.output_for == 'api'):
            if (args.model_type == 'openai'):
                amr_graph = call_openai(args,step_1b_prompt, args.number_output, args.temperature)

                print("STEP 1b: Get AMR Graph", step_1b_prompt)
            else:
                amr_graph = call_palm(args, model, ori_step_1b, sent_input)
                print ("AMR graph", amr_graph)
                if ('\n\n' in amr_graph):
                    amr_graph = amr_graph.split('\n\n')[1]
                amr_graph = amr_graph.replace('```','').strip()
                print ("Output graph", repr(amr_graph))

                print ("STEP 1b context", ori_step_1b)
                print ("STEP 1b input", sent_input)
            
            print ("AMR")
            print (amr_graph)
        else:
            print("STEP 1b: Get AMR Graph", step_1b_prompt)
            amr_graph=''
    else:
        amr_graph = ""

    return amr_graph


def generate_kp(args, selected_demo_dict_ex, domain_name, amr_graph, intent):
    sent_input = ''
    if (args.structure_rep != 'none'):
        # -- Step 2 Demo
        if (args.add_demo == 'true'):
            iter_demo = 0

            demo_2 = gen_step_prompt(args, step_number='2')
            for idx in range (len(selected_demo_dict_ex['utt'])): 
                demo_utt = selected_demo_dict_ex['utt'][idx]
                demo_intent = selected_demo_dict_ex['intent'][idx]
                demo_kp = selected_demo_dict_ex['key_phrase'][idx]
                demo_amr = selected_demo_dict_ex['AMR'][idx]

                demo_domain = selected_demo_dict_ex['domain'][idx]

                demo_2 += 'Sentence: ' + demo_utt + '\n'

                if (args.type_condition != 'none'):
                    demo_2 = condition_intent_info(args, demo_2,demo_intent, intent_map)
                demo_2 += structure_map[args.structure_rep] + ': ' + demo_amr + '\n'
                demo_2 += 'Key phrases: ' + demo_kp + '\n##\n'
                iter_demo += 1

            assert iter_demo == args.number_demo

            step_2_prompt  = demo_2 + '\n' 
        else:
            step_2_prompt = gen_step_prompt(args, step_number='2')

        ori_step_2 = step_2_prompt

        step_2_prompt += '\n Sentence: ' + utterance + '\n'
        sent_input = 'Sentence: ' + utterance + '\n'
        if (args.type_condition != 'none'):
            step_2_prompt = condition_intent_info(args, step_2_prompt,intent, intent_map)
            sent_input = condition_intent_info(args, sent_input,intent, intent_map)

        step_2_prompt += structure_map[args.structure_rep] + ': ' + amr_graph + '\n'

        sent_input += structure_map[args.structure_rep] + ': ' + amr_graph + '\n'

    else: # No structure (for Ablation) - No demos
        step_2_prompt = gen_step_2_nostruct + 'Sentence: ' + utterance + '\n'
    if (args.add_domain == 'true'):
        step_2_prompt += 'Domain name: ' +domain_name + '\n'
        sent_input += 'Domain name: ' + domain_name + '\n'
    step_2_prompt += '\n Key phrases: \n'

    sent_input += 'Key phrases: \n'

    if (args.output_for == 'api'):
        if (args.model_type == 'openai'):
            key_phrases = call_openai(args,step_2_prompt, args.number_output, args.temperature)
            print("STEP 2: Keyphrase Prompt")
            print (step_2_prompt)
        else:

            key_phrases = call_palm(args, model, ori_step_2, sent_input)
            print ("STEP 2 context", ori_step_2)
            print ("STEP 2 input", sent_input)
            
            if ('\n\n' in key_phrases):
                key_phrases = key_phrases.split('\n\n')[1]

        print("OUTPUT STEP 2: KEYPHRASES")
        print (key_phrases)

    else:
        print("STEP 2: Get Key Phrases", step_2_prompt)
        key_phrases=''
    return key_phrases



def generate_slot_label(args, selected_demo_dict_ex, domain_name, amr_graph, intent,key_phrases):
    if (args.add_demo == 'true'):  
        demo_3 = gen_step_prompt(args, step_number='3')
        
        iter_demo = 0
        for idx in range (len(selected_demo_dict_ex['utt'])): 
            demo_utt = selected_demo_dict_ex['utt'][idx]
            demo_intent = selected_demo_dict_ex['intent'][idx]
            demo_kp = selected_demo_dict_ex['key_phrase'][idx]
            demo_pair = selected_demo_dict_ex['pair'][idx]

            demo_3 += 'Sentence: ' + demo_utt + '\n'
            if (args.type_condition != 'none'):
                demo_3 = condition_intent_info(args, demo_3,demo_intent, intent_map)
            if (args.structure_rep != 'none'):
                demo_3 += structure_map[args.structure_rep] + ': ' + demo_amr + '\n'

            demo_3 += 'Key phrases: ' + demo_kp + '\n'


            demo_domain = selected_demo_dict_ex['domain'][idx]
            demo_3 += 'Domain Name: ' + demo_domain + '\n'

            demo_3 += 'List of (Slot Type, Key phrase) pairs separated by commas : ' + demo_pair + '\n##\n'
            iter_demo += 1

        assert iter_demo == args.number_demo
        step_3_prompt  = demo_3 + '\n' 
    else:
        step_3_prompt = gen_step_prompt(args, step_number='3')
    ori_step_3 = step_3_prompt
    sent_input = ''
    step_3_prompt += 'Sentence: ' + utterance + '\n'

    sent_input = 'Sentence: ' + utterance + '\n'
    if (args.type_condition != 'none'):
        step_3_prompt = condition_intent_info(args, step_3_prompt,intent, intent_map)
        sent_input = condition_intent_info(args, sent_input,intent, intent_map)
        if (args.structure_rep != 'none'):
            step_3_prompt += structure_map[args.structure_rep] + ': ' + amr_graph + '\n'
            sent_input += structure_map[args.structure_rep] + ': ' + amr_graph + '\n'

    step_3_prompt += 'Key phrases: ' + key_phrases + '\n'
    sent_input += 'Key phrases: ' + key_phrases + '\n'
    if (args.add_domain == 'true'):
        step_3_prompt += 'Domain name: ' + domain_name + '\n'

        sent_input += 'Domain name: ' + domain_name + '\n'
    step_3_prompt += 'Slot Type, Key phrase  pairs: \n'
    if (args.output_for == 'api'):
        if (args.model_type == 'openai'):
            slot_type = call_openai(args,step_3_prompt, args.number_output, args.temperature)
            print("STEP 3: Get Slot Type", step_3_prompt)
            print ("STEP 3: Slot Type: ")
            print (slot_type)
        else:
            slot_type = call_palm(args, model, ori_step_3, sent_input)
            if ('\n\n' in slot_type):
                slot_type = slot_type.split('\n\n')[1]

            print ("STEP 3 context", ori_step_3)
            print ("STEP 3 input", sent_input)

            print ("---OUTPUT SLOT TYPE")
            print(slot_type)

    return slot_type

def generate_lf(args, selected_demo_dict_ex, domain_name, amr_graph, intent,key_phrases, slot_type):
    if (args.add_demo == 'true'):

        demo_4 = gen_step_prompt(args, step_number='4')
        iter_demo=0
        for idx in range (len(selected_demo_dict_ex['utt'])): 

            demo_utt = selected_demo_dict_ex['utt'][idx]
            demo_intent = selected_demo_dict_ex['intent'][idx]
            demo_kp = selected_demo_dict_ex['key_phrase'][idx]
            demo_pair = selected_demo_dict_ex['pair'][idx]
            demo_lf = selected_demo_dict_ex['lf'][idx]

            demo_4 += 'Sentence: ' + demo_utt + '\n'
            demo_4 = condition_intent_info(args, demo_4,demo_intent, intent_map)
            demo_4 += "Slot Type, Slot Value pairs: " + demo_pair + '\n'
            demo_domain = selected_demo_dict_ex['domain'][idx]
            demo_4 += 'Domain Name: ' + demo_domain + '\n'
            demo_4 += 'Logic Form: ' + demo_lf + '\n##\n'
            iter_demo += 1

        assert iter_demo == args.number_demo

        step_4_prompt  = demo_4 + '\n'
    else:
        step_4_prompt = gen_step_prompt(args, step_number='4')
    
    ori_step_4 = step_4_prompt
    sent_input =''
    step_4_prompt += 'Sentence: ' + utterance + '\n'

    sent_input = 'Sentence: ' + utterance + '\n'

    step_4_prompt += "Intent: " + intent + '\n'
    step_4_prompt += "Slot Type, Slot Value pairs: " + slot_type + '\n'


    sent_input += "Intent: " + intent + '\n'
    sent_input += "Slot Type, Slot Value pairs: " + slot_type + '\n'

    if (args.add_domain == 'true'):
        step_4_prompt += 'Domain name: ' + domain_name + '\n'
        sent_input += 'Domain name: ' + domain_name + '\n'
    step_4_prompt += 'Logic Form: \n'
    sent_input += 'Logic Form: \n'

    if (args.output_for == 'api'):
        if (args.model_type=='openai'):
            pred_lf = call_openai(args,step_4_prompt, args.number_output, args.temperature)
            print("STEP 4: Get Logic Form", step_4_prompt)
        else:
            pred_lf = call_palm(args, model, ori_step_4, sent_input)
            print ("STEP 4 context", ori_step_4)
            print ("STEP 4 input", sent_input)

            pred_lf = reformat_output(pred_lf, slot_vocab)
            pred_lf = pred_lf.split('```')[-1]
            pred_lf = pred_lf.split('```')[0].strip()


        print ("FINAL OUT LF ", pred_lf)
        print ("Target", logical_form)
    else:
        pred_lf =''
        print("STEP 4: Get Logic Form", step_4_prompt)

 
result = []
for example in open(input_test_file, 'r'):
    if (args.dataset =='MTOP'):
        utterance, logical_form, _, domain_name, tag = example.strip().split("\t")
    else:
        utterance, logical_form,domain_name, tag = example.strip().split("\t")
    print ("utt", utterance, logical_form)

    amr_graph=""
    intent=""
    kp=""
    slot_type=""

    # --- Step 1a: Get Intent
    # --- Demo Step 1a
    print ("===========================================================================")
    amr_graph = generate_amr(args, selected_demo_dict_ex, domain_name)
    print ("===========================================================================")
    intent = gen_intent_prompt(args, selected_demo_dict_ex, domain_name)
    print ("===========================================================================")
    # --- Step 2: Get Key Phrases
    kp = generate_kp(args, selected_demo_dict_ex, domain_name, amr_graph, intent)
    print ("===========================================================================")
    slot_type = generate_slot_label(args, selected_demo_dict_ex, domain_name, amr_graph, intent,kp)
    print ("===========================================================================")
    pred_lf=generate_lf(args, selected_demo_dict_ex, domain_name, amr_graph, intent,key_phrases, slot_type)
    result.append({"utterance": utterance, "intent": intent, "AMR Graph": amr_graph, "key_phrase":
        key_phrases, "slot_type": slot_type, "pred_lf": pred_lf, "gold_lf": logical_form})

print ("Output file name",  result_output_file)
if (args.write_output == 'true'):
    json.dump(result, writer, indent=4)
