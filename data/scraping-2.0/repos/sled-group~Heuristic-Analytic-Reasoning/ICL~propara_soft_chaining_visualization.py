import sys
sys.path.append("../")

import json
import numpy as np
from numpy.linalg import norm
import re
from argparse import ArgumentParser
import os
import datetime
from pprint import pprint
from tqdm import tqdm
import time
import openai
import pickle
# from llama_cpp import Llama
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from fastchat.model import load_model

from data.propara import story_pair_prompt_generator, top_down_demo_full, bottom_up_demo_full, load_propara_dataset
from eval.propara import check_response, response_extractor
from models import get_chat_message, API_COSTS, prompt_gpt3_with_caching, prompt_llama_with_caching, prompt_chat_gpt_with_caching, prompt_fastchat_with_caching, VICUNA13B_PATH, ALPACA13B_PATH
from utils import get_output_dir_name
from visualization import separation_generator

def remove_empty_strings(my_list):
    return [item for item in my_list if item != ""]

# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-03-29 16:10:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-12 09:56:12


## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.

def generate_heatmap_upon_text(text_list, attention_list, latex_file, color='red'):
    assert(len(text_list) == len(attention_list))
    word_num = len(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth]{standalone}
            \special{papersize=210mm,297mm}
            \usepackage{color}
            \usepackage{tcolorbox}
            \usepackage{CJK}
            \usepackage{adjustbox}
            \tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
            \begin{document}
            \begin{CJK*}{UTF8}{gbsn}'''+'\n')
        string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
        for idx in range(word_num):
            if text_list[idx] == ' ':
                formatted_number = "{:.10f}".format(attention_list[idx])
                string += "\\colorbox{%s!%s}{"%(color, formatted_number)+"\\strut \hspace{0.05em}" + text_list[idx]+"}"
            elif text_list[idx] == '\n':
                string += r"\newline"
            else:
                formatted_number = "{:.10f}".format(attention_list[idx])
                string += "\\colorbox{%s!%s}{"%(color, formatted_number)+"\\strut " + text_list[idx]+"}"
        string += "\n}}}"
        f.write(string+'\n')
        f.write(r'''\end{CJK*}\end{document}''')

def slice_4d_list(lst, d2_idx):
    """
    Slice a 4D list with inhomogeneous 4th dimension (list of lists of lists of lists).

    Parameters:
    lst (list): the list to slice.
    d2_idx (int): the index for the second dimension.

    Returns:
    list: the sliced list.
    """
    return [[[sublist3 for sublist3 in sublist2[d2_idx]] for sublist2 in sublist1] for sublist1 in lst]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lm_backbone", default="llama7b", choices=["gpt3", "gpt4", "chatgpt", "alpaca13b", "vicuna13b", "llama7b", "llama13b", "llama30b", "llama65b"])
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--local_only", action="store_true", default=False)
    parser.add_argument("--demo_choice", default="stories-4", choices=["stories-4", "balanced-6"])
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--cache_only", action="store_true", default=False)

    parser.add_argument("--action_type", type=str, default="conversion", choices=["conversion"])

    parser.add_argument("--reasoning_depth", choices=["accurate", "consistent", "verifiable"], default='verifiable')
    parser.add_argument("--reasoning_direction", choices=["top-down"], default='top-down')
    parser.add_argument("--output_attn", action="store_true", default=False)

    parser.add_argument("--output_dir", type=str, default="saved_results")
    parser.add_argument("--debug", action='store_true', help="Whether to run the model on only a small amount of data.")
    parser.add_argument("--skip_prompting", action='store_true', help="Whether to NOT prompt any LMs, just generate prompts and inspect them.")
    args = parser.parse_args()

    LM_BACKBONE_ENGINES = {'gpt4': 'gpt-4', 'chatgpt': 'gpt-35-turbo', 'alpaca13b': 'alpaca13b'}

    action_types = [args.action_type]
    dataset = load_propara_dataset(action_types=action_types)
    train_dataset = dataset[args.action_type]["train"]
    test_dataset = dataset[args.action_type]["test"]
    if args.debug:
        test_dataset = test_dataset[:5]

    # CACHE_FNAME = "/home/sstorks/data/sstorks/MultimodalImaginationForPhysicalCausality/cache_files/trip_lm_cache_gpt4_verifiable_top-down.pkl"
    CACHE_FNAME = f'cache_files/propara_lm_cache_{args.lm_backbone}_{args.demo_choice}_{args.reasoning_depth}_{args.reasoning_direction}.pkl'
    print("Using CACHE_FNAME:", CACHE_FNAME)
    if os.path.exists(CACHE_FNAME):
        lm_cache = pickle.load(open(CACHE_FNAME, 'rb'))
    else:
        lm_cache = {}
    args.cache_fname = CACHE_FNAME

    TOKEN_COUNTS = {'prompt': 0, 'gen': 0}


    if args.lm_backbone in ["vicuna13b"]:
        print("Setting up Vicuna...")
        if not args.skip_prompting:
            model, tokenizer = load_model(
                VICUNA13B_PATH,
                "cuda",
                torch.cuda.device_count(),
                load_8bit=False,
            )
        else:
            model, tokenizer = None, None

    if args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
        print("Setting up LLaMA...")
        if not args.skip_prompting and not args.cache_only:
            if args.model_path is None and args.local_only:
                raise ValueError("Must specify --model_path if --local_only is set.")
            model_path = args.model_path if args.model_path is not None else f"decapoda-research/llama-{args.lm_backbone[-2:]}-hf"
            local_only = args.local_only
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=local_only,
            )
            print("Loaded tokenizer.")
            config = AutoConfig.from_pretrained(model_path)
            # reconfigure to output attentions
            config.output_attentions = args.output_attn
            config.return_dict_in_generate = args.output_attn
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
            model.tie_weights()
            model = load_checkpoint_and_dispatch(
                model, 
                model_path, 
                device_map="auto", 
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=torch.float16,
            )
            print("Loaded model.")
        else:
            model, tokenizer = None, None

    
    if args.lm_backbone in ["gpt4", "chatgpt", "vicuna13b"]:
        preamble = "You are a smart chatbot capable of physical commonsense reasoning, and understanding the physical state of the world implied in written texts. "
        preamble += "You will read stories that involve physical actions, and will be asked to "
        
        # Generate preamble message for chat models
        if args.reasoning_depth == 'accurate':
            # If aiming for accurate reasoning, we don't need to care about reasoning direction (there's only one step)
            preamble += (
                "determine which of two stories is more plausible by comparing possible conflicts within them. "
            )
        else:
            # Otherwise, build the description of the task in the appropriate order
            if args.reasoning_direction == "bottom-up":
                # For bottom-up mode, task is posed as physical states -> conflicts -> plausibility
                if args.reasoning_depth == "verifiable":
                    # If targeting verifiable reasoning, describe physical state prediction then conflict detection
                    preamble += (
                        "report the physical states of entities in each story by answering questions about them before and after different actions are taken in the story. "
                        "The story may not explicitly mention state changes. In these cases, you have to use commonsense reasoning to infer the state changes. "
                        "However, you will be penalized if you incorrectly predict an entity's physical state, so you should only predict states you're absolutely sure of based on the text. "
                        "You will then use your understanding of the physical state changes in each story to locate a plausibility conflict. "
                        "Lastly, you will determine which of two stories is more plausible by comparing possible conflicts within them. "
                    )
                elif args.reasoning_depth == "consistent":
                    # If targeting consistent reasoning, only describe conflict detection
                    preamble += (
                        "use your understanding of the physical state changes in each story to locate a plausibility conflict. "
                        "Lastly, you will determine which of two stories is more plausible by comparing possible conflicts within them. "
                    )
            elif args.reasoning_direction == "top-down":
                # For top-down mode, task is posed as plausibility -> conflicts -> physical states
                if args.reasoning_depth == "verifiable":
                    # If targeting verifiable reasoning, describe physical state prediction then conflict detection
                    preamble += (
                        "determine which of two stories is more plausible by comparing possible conflicts within them. "
                        "You will then use your understanding of the physical state changes in each story to locate a plausibility conflict. "
                        "Lastly, you will report the physical states of entities in each story by answering questions about them before and after different actions are taken in the story. "
                        "The story may not explicitly mention state changes. In these cases, you have to use commonsense reasoning to infer the state changes. "
                        "However, you will be penalized if you incorrectly predict an entity's physical state, so you should only predict states you're absolutely sure of based on the text. "
                    )
                elif args.reasoning_depth == "consistent":
                    # If targeting consistent reasoning, only describe conflict detection
                    preamble += (
                        "determine which of two stories is more plausible by comparing possible conflicts within them. "
                        "Lastly, you will use your understanding of the physical state changes in each story to locate a plausibility conflict. "
                    )


    # Demos for action precondition prediction (APP) and action effect prediction (AEP)
    # app_demos = generate_app_demos(train_dataset)
    # aep_demos = generate_aep_demos(train_dataset)

    if args.demo_choice == "stories-4":
        selected_story_example_ids = [200, 68, 107, 311]
    # elif args.demo_choice == "balanced-6":
    #     # sample 2 stories for 3 types of conflict
    #     selected_story_example_ids = balanced_sample_story_ids(train_dataset, select_n_stories_for_each_category=2) # If encounters any error, change a random seed!
    else:
        raise Exception("demo_choice out of options")
    
    print("selected story IDs:", selected_story_example_ids)

    story_demos = ""
    for selected_story_id in selected_story_example_ids:
        story_pair = train_dataset[selected_story_id]
        if args.reasoning_direction == 'top-down':
            story_demos += top_down_demo_full(story_pair) + '\n'
        elif args.reasoning_direction == 'bottom-up':
            story_demos += bottom_up_demo_full(story_pair) + '\n'
        else:
            raise Exception(f"reasoning_direction: {args.reasoning_direction} not supported yet")
    
    # print(story_demos)

   # Start testing
    predictions = []
    plausibility_score, conflict_score, physical_states_score = 0, 0, 0

    for i in range(0, len(test_dataset)):
        example_id = test_dataset[i]['example_id']
        prediction_obj = {'example_id': example_id}
        story_pair = test_dataset[i]
        if story_pair['story_converted'] == 'A':
            story_converted = story_pair["story_A_sentences"]
        else:
            story_converted = story_pair["story_B_sentences"]
        story_prompt = story_pair_prompt_generator(story_pair)

        if args.lm_backbone in ['gpt3']:
            prompt = ""
            prompt += story_demos + "\n" + story_prompt
            prediction_obj['input_prompt'] = prompt.split('\n')
            max_tokens=128
            story_generated_text = prompt_gpt3_with_caching(prompt, args, lm_cache, example_id, max_tokens=max_tokens, token_counts=TOKEN_COUNTS)

        if args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
            if args.output_attn:
                separations = separation_generator(story_demos + "\n", story_prompt, tokenizer) 
                separations[0][1] = separations[0][1][:-1]
                separations[1][1][-1] = separations[0][1][-1]
            else:
                separations = None
            prompt = ""
            prompt += story_demos + "\n" + story_prompt
            prediction_obj['input_prompt'] = prompt.split('\n')
            max_tokens=128
            story_generated_text = prompt_llama_with_caching(model, tokenizer, prompt, args, lm_cache, str(example_id), max_tokens=max_tokens, output_attn=args.output_attn, separations=separations)

        elif args.lm_backbone in ['gpt4', "chatgpt", "vicuna13b"]:
            # Generate chat model messages           
            preamble_message = get_chat_message("system", preamble, model_name=args.lm_backbone)
            story_demos_message = (
                get_chat_message(
                    "system",
                    "And here are some examples of how you can use your understanding to solve the two-story task that the user will give you:\n\n" + story_demos,
                    model_name=args.lm_backbone
                )
            )
            # Compile messages based on passed in arguments
            messages = [preamble_message]
            messages += [story_demos_message]
            messages += [
                get_chat_message(
                    "user",
                    story_prompt,
                    model_name=args.lm_backbone
                )
            ]
            if args.lm_backbone in ['gpt4', 'chatgpt']:
                prediction_obj['input_prompt'] = [get_chat_message(
                                                    message['role'], 
                                                    message['content'].split('\n'),
                                                    model_name=args.lm_backbone,
                                                ) for message in messages]
            elif args.lm_backbone in ['vicuna13b']:
                prediction_obj['input_prompt'] = [get_chat_message(
                                                    message[0], 
                                                    message[1].split('\n'),
                                                    model_name=args.lm_backbone,
                                                ) for message in messages]
            
            if args.lm_backbone in ['gpt4', 'chatgpt']:
                story_generated_text = prompt_chat_gpt_with_caching(messages, args, lm_cache, example_id, max_tokens=1024 if args.reasoning_direction == 'bottom-up' else 128, token_counts=TOKEN_COUNTS)
            elif args.lm_backbone in ['vicuna13b']:
                story_generated_text = prompt_fastchat_with_caching(model, tokenizer, messages, args, lm_cache, example_id, max_tokens=1024 if args.reasoning_direction == 'bottom-up' else 128)
        prediction_obj['generated_text'] = story_generated_text.split('\n')
        if args.reasoning_direction == 'top-down':
            response = response_extractor(story_generated_text, type='top_down')
            prediction_obj['response'] = response['top_down']
            acc, cons, ver = check_response(story_generated_text, story_pair, demo_type='top_down')
            plausibility_score += acc
            conflict_score += cons
            physical_states_score += ver
            if (acc, cons, ver) == (1, 1, 1):
                pass
        elif args.reasoning_direction == 'bottom-up':
            response = response_extractor(story_generated_text, type='bottom_up')
            prediction_obj['response'] = response['bottom_up']
            acc, cons, ver = check_response(story_generated_text, story_pair, demo_type='bottom_up')
            plausibility_score += acc
            conflict_score += cons
            physical_states_score += ver
            if (acc, cons, ver) == (1, 1, 1):
                pass
        else:
            raise NotImplementedError("Bottom-up reasoning not implemented yet")   
        predictions.append(prediction_obj)
        correct_story, correct_sentence, correct_physical_states = acc, cons, ver
        all_lines_shorter_than_50char = True
        for line in story_prompt.split('\n'):
            if len(line) > 70:
                all_lines_shorter_than_50char = False
        if correct_story and correct_sentence and correct_physical_states:
            print("Selected ID:", i)

    plausibility_score, conflict_score, physical_states_score = 0, 0, 0
    correct_story_total_attn = 0
    correct_sentence_total_attn = 0
    incorrect_story_total_attn = 0
    incorrect_sentence_total_attn = 0
    num_correct_story = 0
    num_correct_sentence = 0

    story_attn_threshold = 0.1
    num_story_attn_exceed_threshold = 0
    sentence_attn_threshold = 0.1
    num_sentence_attn_exceed_threshold = 0

    num_true_positive_second_level = 0
    num_true_negative_second_level = 0
    num_false_positive_second_level = 0
    num_false_negative_second_level = 0
    num_true_positive_third_level = 0
    num_true_negative_third_level = 0
    num_false_positive_third_level = 0
    num_false_negative_third_level = 0

    # i = 32 # pick one (second level)
    i = 56 # pick one (third level)

    print("participant converted:", test_dataset[i]['participant_converted'])
    print("conversion:", test_dataset[i]['conversions'])

    example_id = test_dataset[i]['example_id']

    prediction_obj = {'example_id': example_id}
    story_pair = test_dataset[i]
    if story_pair['story_converted'] == 'A':
        story_converted = story_pair["story_A_sentences"]
    else:
        story_converted = story_pair["story_B_sentences"]
    story_prompt = story_pair_prompt_generator(story_pair)

    # Prompt the LM
    if args.lm_backbone in ['gpt3']:
        prompt = ""
        prompt += story_demos + "\n" + story_prompt
        # Record prompt used with LM (splitting by newlines for readability)
        prediction_obj['input_prompt'] = prompt.split('\n')
        max_tokens=128
        story_generated_text = prompt_gpt3_with_caching(prompt, args, lm_cache, example_id, max_tokens=max_tokens, token_counts=TOKEN_COUNTS)

    if args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
        if args.output_attn:
            separations = separation_generator(story_demos + "\n", story_prompt, tokenizer) 
            separations[0][1] = separations[0][1][:-1]
            separations[1][1][-1] = separations[0][1][-1]
        else:
            separations = None
        prompt = ""
        prompt += story_demos + "\n" + story_prompt
        # Record prompt used with LM (splitting by newlines for readability)
        prediction_obj['input_prompt'] = prompt.split('\n')
        max_tokens=128
        story_generated_text = prompt_llama_with_caching(model, tokenizer, prompt, args, lm_cache, str(example_id), max_tokens=max_tokens, output_attn=args.output_attn, separations=separations)
        # if args.output_attn:
        #     attn = lm_cache[str(example_id)+'_attn']

    elif args.lm_backbone in ['gpt4', "chatgpt", "vicuna13b"]:
        # Generate chat model messages           
        preamble_message = get_chat_message("system", preamble, model_name=args.lm_backbone)

        story_demos_message = (
            get_chat_message(
                "system",
                "And here are some examples of how you can use your understanding to solve the two-story task that the user will give you:\n\n" + story_demos,
                model_name=args.lm_backbone
            )
        )

        # Compile messages based on passed in arguments
        messages = [preamble_message]

        # # To fully specify physical state prediction step, prepend several demos of it covering all physical attributes first
        # if args.reasoning_depth == "verifiable":
            # Add some demos of APP and AEP specifically
            # app_demos_message = (
            #     get_chat_message(
            #         "system",
            #         "Here are some examples of how you can understand physical *precondition* states in text:\n\n" + app_demos,
            #         model_name=args.lm_backbone
            #     )                
            # )
            # aep_demos_message = (
            #     get_chat_message(
            #         "system",
            #         "Here are some examples of how you can understand physical *effect* states in text:\n\n" + aep_demos,
            #         model_name=args.lm_backbone
            #     )
            # )
            # messages += [app_demos_message, aep_demos_message]

        messages += [story_demos_message]
        messages += [
            get_chat_message(
                "user",
                story_prompt,
                model_name=args.lm_backbone
            )
        ]
        # Record prompt used with LM (splitting all messages by newlines for readability)
        if args.lm_backbone in ['gpt4', 'chatgpt']:
            prediction_obj['input_prompt'] = [get_chat_message(
                                                message['role'], 
                                                message['content'].split('\n'),
                                                model_name=args.lm_backbone,
                                            ) for message in messages]
        elif args.lm_backbone in ['vicuna13b']:
            prediction_obj['input_prompt'] = [get_chat_message(
                                                message[0], 
                                                message[1].split('\n'),
                                                model_name=args.lm_backbone,
                                            ) for message in messages]
        
        if args.lm_backbone in ['gpt4', 'chatgpt']:
            story_generated_text = prompt_chat_gpt_with_caching(messages, args, lm_cache, example_id, max_tokens=1024 if args.reasoning_direction == 'bottom-up' else 128, token_counts=TOKEN_COUNTS)
        elif args.lm_backbone in ['vicuna13b']:
            story_generated_text = prompt_fastchat_with_caching(model, tokenizer, messages, args, lm_cache, example_id, max_tokens=1024 if args.reasoning_direction == 'bottom-up' else 128)

    # Gather up model predictions using templates and regular expressions
    prediction_obj['generated_text'] = story_generated_text.split('\n')
    # print("generated text:", prediction_obj['generated_text'])
    if args.reasoning_direction == 'top-down':
        response = response_extractor(story_generated_text, type='top_down')
        prediction_obj['response'] = response['top_down']
        acc, cons, ver = check_response(story_generated_text, story_pair, demo_type='top_down')
        plausibility_score += acc
        conflict_score += cons
        physical_states_score += ver
        if (acc, cons, ver) == (1, 1, 1):
            # print("Correct!")
            pass
    elif args.reasoning_direction == 'bottom-up':
        response = response_extractor(story_generated_text, type='bottom_up')
        prediction_obj['response'] = response['bottom_up']
        acc, cons, ver = check_response(story_generated_text, story_pair, demo_type='bottom_up')
        plausibility_score += acc
        conflict_score += cons
        physical_states_score += ver
        if (acc, cons, ver) == (1, 1, 1):
            # print("Correct!")
            pass
    else:
        raise NotImplementedError("Bottom-up reasoning not implemented yet")   
    predictions.append(prediction_obj)
    correct_story, correct_sentence, correct_physical_states = acc, cons, ver
    if correct_story:
        num_correct_story += 1
        all_attn = lm_cache[str(example_id) + "_attn"]
        story_A_attn = np.array(slice_4d_list(all_attn, 0))
        story_B_attn = np.array(slice_4d_list(all_attn, 1))
        story_A_attn_mean = np.mean(story_A_attn[30:40], axis=0)
        story_B_attn_mean = np.mean(story_B_attn[30:40], axis=0)
        if story_pair['story_converted'] == 'A':
            second_level_converted_story_attn_mean = np.mean(story_A_attn_mean[1])
            second_level_irrelevant_story_attn_mean = np.mean(story_B_attn_mean[1])
            third_level_converted_story_attn = story_A_attn_mean[2]
        else:
            second_level_converted_story_attn_mean = np.mean(story_B_attn_mean[1])
            second_level_irrelevant_story_attn_mean = np.mean(story_A_attn_mean[1])
            third_level_converted_story_attn = story_B_attn_mean[2]
        if second_level_converted_story_attn_mean > story_attn_threshold:
            num_story_attn_exceed_threshold += 1
            if correct_sentence:
                num_true_positive_second_level += 1
            else:
                num_false_positive_second_level += 1
        else:
            if correct_sentence:
                num_false_negative_second_level += 1
            else:
                num_true_negative_second_level += 1
        correct_story_total_attn += second_level_converted_story_attn_mean
        incorrect_story_total_attn += second_level_irrelevant_story_attn_mean
        sentence_index = story_pair["conversions"][0]["state_converted_to"] - 1
        third_level_converted_sentence_attn = third_level_converted_story_attn[sentence_index]
        third_level_irrelevant_sentences_attn_mean = mean = np.mean(np.delete(third_level_converted_story_attn, sentence_index))
        if correct_sentence:
            print("2nd level correct")
            num_correct_sentence += 1
            if third_level_converted_sentence_attn > sentence_attn_threshold:
                num_sentence_attn_exceed_threshold += 1
                if correct_physical_states:
                    num_true_positive_third_level += 1
                else:
                    num_false_positive_third_level += 1
            else:
                if correct_physical_states:
                    num_false_negative_third_level += 1
                else:
                    num_true_negative_third_level += 1
            correct_sentence_total_attn += third_level_converted_sentence_attn
            incorrect_sentence_total_attn += third_level_irrelevant_sentences_attn_mean
        
        story_A_attention_second_level_scaled = (np.mean(story_A_attn_mean[1]) / (np.mean(story_A_attn_mean[1]) + np.mean(story_B_attn_mean[1]))) * 100
        story_B_attention_second_level_scaled = (np.mean(story_B_attn_mean[1]) / (np.mean(story_A_attn_mean[1]) + np.mean(story_B_attn_mean[1]))) * 100

        sentences_attention_third_level = np.concatenate((story_A_attn_mean[2], story_B_attn_mean[2]))

        sentences_attention_third_level_scaled = sentences_attention_third_level * 100

        story_pair["story_A_sentences"] = remove_empty_strings(story_pair["story_A_sentences"])
        story_pair["story_B_sentences"] = remove_empty_strings(story_pair["story_B_sentences"])

        story_pair_lines = story_prompt.split('\n')
        for i in range(0, len(story_pair_lines)):
            if story_pair_lines[i] == "1. PLants have roots.":
                story_pair_lines[i] = "1. Plants have roots."
        if story_pair_lines[-1] == '':
            story_pair_lines = story_pair_lines[:-1]
        if story_pair['story_converted'] == 'A':
            story_converted_identifier = 'A'
        else:
            story_converted_identifier = 'B'
        attention_list_second_level = []
        attention_list_third_level = []
        context_string = ""
        for char in story_pair_lines[0]:
            context_string += char
            attention_list_second_level.append(0) # account for story A identifier
            attention_list_third_level.append(0) # account for story A identifier
        context_string += '\n'
        attention_list_second_level.append(0) # account \n
        attention_list_third_level.append(0) # account \n
        for i in range(1, len(story_pair["story_A_sentences"]) + 1):
            for char in story_pair_lines[i]:
                context_string += char
                attention_list_second_level.append(story_A_attention_second_level_scaled)
                attention_list_third_level.append(sentences_attention_third_level_scaled[i - 1])
                sentences_attention_third_level_scaled
            context_string += '\n'
            attention_list_second_level.append(0) # account \n
            attention_list_third_level.append(0) # account \n
        for char in story_pair_lines[len(story_pair["story_A_sentences"]) + 1]:
            context_string += char
            attention_list_second_level.append(0) # account for story B identifier
            attention_list_third_level.append(0) # account for story B identifier
        context_string += '\n'
        attention_list_second_level.append(0) # account \n
        attention_list_third_level.append(0) # account \n
        for i in range(2 + len(story_pair["story_A_sentences"]), 2 + len(story_pair["story_A_sentences"]) + len(story_pair["story_B_sentences"])):
            for char in story_pair_lines[i]:
                context_string += char
                attention_list_second_level.append(story_B_attention_second_level_scaled)
                attention_list_third_level.append(sentences_attention_third_level_scaled[i - 2])
            context_string += '\n'
            attention_list_second_level.append(0) # account \n
            attention_list_third_level.append(0) # account \n
        for char in story_pair_lines[2 + len(story_pair["story_A_sentences"]) + len(story_pair["story_B_sentences"])]:
            context_string += char
            attention_list_second_level.append(0) # account for what happened...
            attention_list_third_level.append(0) # account for what happened...

        if not os.path.exists("attention_visualization_results"):
            os.makedirs("attention_visualization_results")
          
        # generate_heatmap_upon_text(context_string, attention_list_second_level, "attention_visualization_results/propara_topdown_second_level_attn_vis.tex", color='red')
        generate_heatmap_upon_text(context_string, attention_list_third_level, "attention_visualization_results/propara_topdown_third_level_attn_vis.tex", color='red')