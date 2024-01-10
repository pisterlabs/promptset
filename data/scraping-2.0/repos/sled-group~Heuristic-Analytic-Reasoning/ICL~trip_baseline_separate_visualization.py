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
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from data.trip import load_trip_dataset
from eval.trip import trip_metrics, trip_metrics_for_one_pred,\
                      story_pair_demo_generator_topdown, story_pair_prompt_generator, conflict_demo_generator_fully_sep, \
                      confl_pairs_extractor_topdown, physical_states_demo_generator_fully_separate, \
                      generate_aep_demos_fully_separate_familiarization, generate_app_demos_fully_separate_familiarization, extract_and_combine_physical_states_topdown, \
                      implausible_story_conflict_reason_generator, balanced_sample_story_ids
from models import get_chat_message, API_COSTS, prompt_gpt3_with_caching, prompt_chat_gpt_with_caching, VICUNA13B_PATH, prompt_llama_with_caching, get_65b_device_mapping
from utils import get_output_dir_name
from visualization import separation_generator

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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lm_backbone", default="gpt3", choices=["gpt3", "gpt4", "chatgpt", "vicuna", "llama7b", "llama13b", "llama30b", "llama65b"])
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--local_only", action="store_true", default=False)
    parser.add_argument("--demo_choice", default="stories-4", choices=["stories-4", "balanced-6"])
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--num_demos", type=int, default=0)

    parser.add_argument("--use_conflict_explanations", action="store_true")
    parser.add_argument("--exclude_multi_confl", default=False, action="store_true")
    parser.add_argument("--condense_multiple_conflict", default=False, action="store_true")
    parser.add_argument("--cache_only", default=False)
    parser.add_argument("--reduce_options", default=True)
    parser.add_argument("--output_attn", action="store_true", default=False)

    parser.add_argument("--output_dir", type=str, default="saved_results")
    parser.add_argument("--debug", action='store_true', help="Whether to run the model on only a small amount of data.")
    parser.add_argument("--skip_prompting", action='store_true', help="Whether to NOT prompt any LMs, just generate prompts and inspect them.")
    args = parser.parse_args()

    LETTER_TO_IDX = {'A': 0, 'B': 1}

    train_dataset = load_trip_dataset(exclude_multi_confl=False, condense_multiple_conflict=False, reduce_options=False)["train"]
    test_dataset = load_trip_dataset(exclude_multi_confl=args.exclude_multi_confl, condense_multiple_conflict=args.condense_multiple_conflict, reduce_options=args.reduce_options)["test"]
    if args.debug:
        test_dataset = test_dataset[:5]

    CACHE_FNAME = f'cache_files/trip_lm_cache_{args.lm_backbone}_{args.demo_choice}_baseline-separate.pkl'
    print("Using CACHE_FNAME:", CACHE_FNAME)
    if args.use_conflict_explanations:
        CACHE_FNAME = CACHE_FNAME.replace('.pkl', '_expl.pkl')
    if os.path.exists(CACHE_FNAME):
        lm_cache = pickle.load(open(CACHE_FNAME, 'rb'))
        print(len(lm_cache))
    else:
        lm_cache = {}
    # print(lm_cache)
    args.cache_fname = CACHE_FNAME

    TOKEN_COUNTS = {'prompt': 0, 'gen': 0}


    if args.lm_backbone in ["gpt4", "chatgpt"]:
        preamble = "You are a smart chatbot capable of physical commonsense reasoning, and understanding the physical state of the world implied in written texts. "
        preamble += "You will read stories that involve physical actions, and will be asked to "
        
        # Generate preamble messages for chat models - one for each separated task
        preamble_plausibility = preamble + (
            "determine which of two stories is more plausible by comparing possible conflicts within them. "
        )
        preamble_conflict = preamble + (
            "use your understanding of the physical state changes in each story to locate a plausibility conflict. "
        )
        preamble_physical_states = preamble + (
            "You will report the physical states of entities in each story by answering questions about them before and after different actions are taken in the story. "
            "The story may not explicitly mention state changes. In these cases, you have to use commonsense reasoning to infer the state changes. "
            "However, you will be penalized if you incorrectly predict an entity's physical state, so you should only predict states you're absolutely sure of based on the text. "
        )
    
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
            if args.lm_backbone == 'llama65b' and args.output_attn:
                device_map = get_65b_device_mapping(config, torch.cuda.device_count(), config.num_hidden_layers//(torch.cuda.device_count()-1)+1)
            else:
                device_map = "auto"
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
            model.tie_weights()
            model = load_checkpoint_and_dispatch(
                model, 
                model_path, 
                device_map=device_map, 
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=torch.float16,
            )
            print("Loaded model.")
        else:
            model, tokenizer = None, None

    selected_story_indexes = None
    if args.demo_choice == "stories-4":
        # selected_story_indexes = [588, 107, 51, 311]
        selected_story_example_ids = ["train_588",
                                      "train_107",
                                      "train_51",
                                      "train_311"]
    elif args.demo_choice == "balanced-6":
        # sample 2 stories for 3 types of conflict
        selected_story_example_ids = balanced_sample_story_ids(train_dataset, select_n_stories_for_each_category=2) # If encounters any error, change a random seed!
    else:
        raise Exception("demo_choice out of options")
    # selected_story_conflict_reasons = {
    #     588: "the donut was put in the trash in sentence 4, but Mary ate the donut in sentence 5",
    #     107: "in sentence 1 Ann realizes that all of her tools are missing yet in sentence 2 she is taking out an axe, a pair of scissors, rope and a few other things",
    #     0: "Tom put the soup in the microwave and then ate it cold",
    #     51: "Sentence 3 states that the hair dryer would not turn on, while Sentence 4 states that Ann used the hair dryer to dry her hair",
    #     311: "you cannot turn something on that is not plugged in",
    # }

    # Generate some demos for story plausibility
    plausibility_demos = ""
    for sampled_train_data_i, selected_story_id in enumerate(selected_story_example_ids):
        demo_example = [example for example in train_dataset if example['example_id'] == selected_story_id][0]
        story_pair = demo_example['stories']
        plausibility_demos += story_pair_demo_generator_topdown(story_pair, 
                                                                conflict_reason_enabled=True if args.use_conflict_explanations else False,
                                                                reasoning_depth="accurate") + "\n"

    # Generate some demos for conflict detection
    conflict_demos = ""
    for sampled_train_data_i, selected_story_id in enumerate(selected_story_example_ids):
        demo_example = [example for example in train_dataset if example['example_id'] == selected_story_id][0]
        story_pair = demo_example['stories']
        story_A = story_pair[0]
        story_B = story_pair[1]
        if story_A["plausible"] == True:
            implausible_story = story_B
            implausible_story_letter = "B"
        elif story_B["plausible"] == True:
            implausible_story = story_A
            implausible_story_letter = "A"
        conflict_demos += story_pair_prompt_generator(story_A, story_B)
        sentence_numbers = (implausible_story['confl_pairs'][0][0] + 1, implausible_story['confl_pairs'][0][1] + 1)
        conflict_demo_temp = conflict_demo_generator_fully_sep(sentence_numbers, implausible_story_letter,
                                                     conflict_reason=implausible_story_conflict_reason_generator(implausible_story) if args.use_conflict_explanations else None)
        # conflict_demo_temp = conflict_demo_temp if args.use_conflict_explanations else conflict_demo_temp.capitalize()
        conflict_demos += conflict_demo_temp + "\n"

    # Demos for action precondition prediction (APP) and action effect prediction (AEP)
    app_demos = generate_app_demos_fully_separate_familiarization(train_dataset, reduce_options=args.reduce_options)
    aep_demos = generate_aep_demos_fully_separate_familiarization(train_dataset, reduce_options=args.reduce_options)

    # Generate some demos for physical states
    physical_states_demos = app_demos + "\n" + aep_demos
    # physical_states_demos = ""
    for sampled_train_data_i, selected_story_id in enumerate(selected_story_example_ids):
        demo_example = [example for example in train_dataset if example['example_id'] == selected_story_id][0]
        story_pair = demo_example['stories']
        story_A = story_pair[0]
        story_B = story_pair[1]
        if story_A["plausible"] == True:
            implausible_story = story_B
            implausible_story_letter = "B"
        elif story_B["plausible"] == True:
            implausible_story = story_A
            implausible_story_letter = "A"
        physical_states_demos += "\n" + story_pair_prompt_generator(story_A, story_B)
        physical_states_demos += physical_states_demo_generator_fully_separate(implausible_story)

    # print(plausibility_demos)
    # print(conflict_demos)
    # print(physical_states_demos)

    
    # For Debugging Purpose:
    # predicted_plausible_story_letter = plausibility_demos.split(' ')[1][:1]
    # print(predicted_plausible_story_letter)
    # predicted_confl_pair = confl_pairs_extractor_topdown(conflict_demos.split(' '))
    # print(predicted_confl_pair)
    # prediction_obj = {}
    # prediction_obj['confl_pairs'] = predicted_confl_pair
    # predicted_physical_states = extract_and_combine_physical_states_topdown(physical_states_demos,
    #                                                                                             prediction_obj, 
    #                                                                                             implausible_story['entities'],
    #                                                                                             implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][0]],
    #                                                                                             implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][1]],)
    # print(predicted_physical_states)
    
    trials = args.num_demos if args.num_demos > 0 else len(test_dataset)

    predictions = []

    correct_story_total_attn = 0
    incorrect_story_total_attn = 0
    correct_sentences_total_attn = 0
    incorrect_sentences_total_attn = 0
    num_correct_story = 0
    num_correct_sentences = 0

    story_attn_threshold = 0.1
    num_story_attn_exceed_threshold = 0
    sentences_attn_threshold = 0.1
    num_sentences_attn_exceed_threshold = 0

    num_true_positive_second_level = 0
    num_true_negative_second_level = 0
    num_false_positive_second_level = 0
    num_false_negative_second_level = 0
    num_true_positive_third_level = 0
    num_true_negative_third_level = 0
    num_false_positive_third_level = 0
    num_false_negative_third_level = 0

    # i = 55 # pick one (second level)
    i = 123 # pick one (third level)

    acc, cons, ver = 0, 0, 0

    example_id = test_dataset[i]['example_id']

    prediction_obj = {'example_id': example_id}
    stories_info = test_dataset[i]['stories']
    if stories_info[0]['plausible'] == True:
        implausible_story = stories_info[1]
        print("B is implausible")
    else:
        implausible_story = stories_info[0]
        print("A is implausible")

    print("story confl pair:", implausible_story["confl_pairs"][0])

    story_prompt = story_pair_prompt_generator(stories_info[0], stories_info[1])

    # STEP 1: Prompt LM for which story is plausible
    if args.lm_backbone in ['gpt3']:
        plausibility_prompt = plausibility_demos
        plausibility_prompt += "\n" + story_pair_prompt_generator(stories_info[0], stories_info[1])
        prediction_obj['input_prompt_plausibility'] = plausibility_prompt.split('\n')
        plausibility_generated_text = prompt_gpt3_with_caching(plausibility_prompt, args, lm_cache, example_id + '_plausibility', token_counts=TOKEN_COUNTS)
    elif args.lm_backbone in ['gpt4', 'chatgpt']:
        plausibility_prompt = []
        plausibility_prompt.append(get_chat_message('system', preamble_plausibility))
        plausibility_prompt.append(get_chat_message('system', "Here are some examples:\n\n" + plausibility_demos))
        plausibility_prompt.append(get_chat_message('user', story_pair_prompt_generator(stories_info[0], stories_info[1])))
        prediction_obj['input_prompt_plausibility'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in plausibility_prompt]
        plausibility_generated_text = prompt_chat_gpt_with_caching(plausibility_prompt, args, lm_cache, example_id + '_plausibility', token_counts=TOKEN_COUNTS)
    elif args.lm_backbone in ["llama7b", "llama13b", "llama30b", "llama65b"]:
        plausibility_prompt = plausibility_demos
        plausibility_prompt += "\n" + story_pair_prompt_generator(stories_info[0], stories_info[1])
        separations = separation_generator(plausibility_demos + "\n", story_pair_prompt_generator(stories_info[0], stories_info[1]), tokenizer) if args.output_attn else None
        plausibility_generated_text = prompt_llama_with_caching(model, tokenizer, plausibility_prompt, args, lm_cache, example_id + '_plausibility', max_tokens=128, output_attn=args.output_attn, separations=separations)
    if args.lm_backbone in ['gpt3', 'gpt4', 'chatgpt']:
        predicted_plausible_story_letter = plausibility_generated_text.split(' ')[1][:1] # Just take second token as the plausible story
    elif args.lm_backbone in ["llama7b", "llama13b", "llama30b", "llama65b"]:
        predicted_plausible_story_letter = plausibility_generated_text.split(' ')[1]
    predicted_implausible_story = stories_info[1 - LETTER_TO_IDX[predicted_plausible_story_letter]]
    prediction_obj['plausible_story'] = predicted_plausible_story_letter
    prediction_obj['generated_text_plausibility'] = plausibility_generated_text.split('\n')

    # If LM was correct, separately ask for the evidence for the end task prediction
    prediction_obj['confl_pairs'] = []
    prediction_obj['physical_states'] = []
    if stories_info[LETTER_TO_IDX[predicted_plausible_story_letter]]['plausible']:
        # Picked the correct story!
        acc = 1

        # STEP 2: Prompt LM for which sentences are conflicting
        if args.lm_backbone in ['gpt3']:
            conflict_prompt = conflict_demos
            conflict_prompt += "\n" + story_pair_prompt_generator(stories_info[0], stories_info[1])
            prediction_obj['input_prompt_conflict'] = conflict_prompt.split('\n')
            conflict_generated_text = prompt_gpt3_with_caching(conflict_prompt, args, lm_cache, example_id + '_conflict', token_counts=TOKEN_COUNTS)
        elif args.lm_backbone in ['gpt4', 'chatgpt']:
            conflict_prompt = []
            conflict_prompt.append(get_chat_message("system", preamble_conflict))
            conflict_prompt.append(get_chat_message("system", "Here are some examples:\n\n" + conflict_demos))
            conflict_prompt.append(get_chat_message("user", story_pair_prompt_generator(stories_info[0], stories_info[1])))
            prediction_obj['input_prompt_conflict'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in conflict_prompt]
            conflict_generated_text = prompt_chat_gpt_with_caching(conflict_prompt, args, lm_cache, example_id + '_conflict', token_counts=TOKEN_COUNTS)
        elif args.lm_backbone in ["llama7b", "llama13b", "llama30b", "llama65b"]:
            conflict_prompt = conflict_demos
            conflict_prompt += "\n" + story_pair_prompt_generator(stories_info[0], stories_info[1])
            separations = separation_generator(conflict_demos + "\n", story_pair_prompt_generator(stories_info[0], stories_info[1]), tokenizer) if args.output_attn else None
            conflict_generated_text = prompt_llama_with_caching(model, tokenizer, conflict_prompt, args, lm_cache, example_id + '_conflict', max_tokens=128, output_attn=args.output_attn, separations=separations)
        predicted_confl_pair = confl_pairs_extractor_topdown(conflict_generated_text.split(' '))
        prediction_obj['confl_pairs'] = predicted_confl_pair
        prediction_obj['generated_text_conflict'] = conflict_generated_text

        if predicted_confl_pair == implausible_story['confl_pairs']:
            # Picked the correct conflicting sentences!

            cons = 1

            # STEP 3: Prompt LM for physical states in conflicting sentences
            if args.lm_backbone in ['gpt3']:
                physical_states_prompt = physical_states_demos
                physical_states_prompt += "\n" + story_pair_prompt_generator(stories_info[0], stories_info[1])
                prediction_obj['input_prompt_physical_states'] = physical_states_prompt.split('\n')
                physical_states_generated_text = prompt_gpt3_with_caching(physical_states_prompt, args, lm_cache, example_id + '_physical_states', token_counts=TOKEN_COUNTS)
            elif args.lm_backbone in ['gpt4', 'chatgpt']:
                physical_states_prompt = []
                physical_states_prompt.append(get_chat_message("system", preamble_physical_states))
                physical_states_prompt.append(get_chat_message("system", "Here are some examples:\n\n" + physical_states_demos))
                physical_states_prompt.append(get_chat_message("user", story_pair_prompt_generator(stories_info[0], stories_info[1])))
                prediction_obj['input_prompt_aep'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in physical_states_prompt]
                physical_states_generated_text = prompt_chat_gpt_with_caching(physical_states_prompt, args, lm_cache, example_id + '_physical_states', token_counts=TOKEN_COUNTS)    
            elif args.lm_backbone in ["llama7b", "llama13b", "llama30b", "llama65b"]:
                physical_states_prompt = physical_states_demos
                physical_states_prompt += "\n" + story_pair_prompt_generator(stories_info[0], stories_info[1])
                separations = separation_generator(physical_states_demos + "\n", story_pair_prompt_generator(stories_info[0], stories_info[1]), tokenizer) if args.output_attn else None
                physical_states_generated_text = prompt_llama_with_caching(model, tokenizer, physical_states_prompt, args, lm_cache, example_id + '_physical_states', max_tokens=128, output_attn=args.output_attn, separations=separations)  

            prediction_obj['physical_states'] = extract_and_combine_physical_states_topdown(physical_states_generated_text,
                                                                                            prediction_obj, 
                                                                                            implausible_story['entities'],
                                                                                            implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][0]],
                                                                                            implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][1]],)
            prediction_obj['generated_text_physical_states'] = physical_states_generated_text

    predictions.append(prediction_obj)
    # correct_story_total_attn
    # incorrect_story_total_attn
    # correct_sentences_total_attn
    # incorrect_sentences_total_attn
    # num_correct_story
    # num_correct_sentences
    correct_plausibility, correct_confl_pairs, correct_physical_states = trip_metrics_for_one_pred(test_dataset[i], prediction_obj)
    correct_plausibility, correct_confl_pairs = acc, cons
    print(correct_plausibility, correct_confl_pairs, correct_physical_states)
    if correct_plausibility and correct_confl_pairs and (not correct_physical_states):
        print("Selected IDs:", i)
    if correct_plausibility:
        num_correct_story += 1
        all_attn_conflict = np.array(lm_cache[example_id + "_conflict_attn"])
        attn_mean_conflict = np.mean(all_attn_conflict[30:50], axis=0) # mean over attention_layer 19,20,21
        if stories_info[0]['plausible'] == True:
            second_level_implausible_story_attn_mean = np.mean(attn_mean_conflict[0][1]) # first 0: generated sentence for conflict; second 1: the second story -> implausible
            second_level_plausible_story_attn_mean = np.mean(attn_mean_conflict[0][0])
        else:
            second_level_implausible_story_attn_mean = np.mean(attn_mean_conflict[0][0]) # first 0: generated sentence for conflict; second 0: the first story -> implausible
            second_level_plausible_story_attn_mean = np.mean(attn_mean_conflict[0][1])
        if second_level_implausible_story_attn_mean > story_attn_threshold:
            num_story_attn_exceed_threshold += 1
            if correct_confl_pairs:
                num_true_positive_second_level += 1
            else:
                num_false_positive_second_level += 1
        else:
            if correct_confl_pairs:
                num_false_negative_second_level += 1
            else:
                num_true_negative_second_level += 1
        correct_story_total_attn += second_level_implausible_story_attn_mean
        incorrect_story_total_attn += second_level_plausible_story_attn_mean
        if correct_confl_pairs:
            confl_sentence_index_1, confl_sentence_index_2 = implausible_story["confl_pairs"][0]
            # print(lm_cache[example_id + "_physical_states"])
            all_attn_physical_states = np.array(lm_cache[example_id + "_physical_states_attn"])
            attn_mean_physical_states = np.mean(all_attn_physical_states[30:50], axis=0)
            if stories_info[0]['plausible'] == True:
                third_level_effect_implausible_story_attn = attn_mean_physical_states[0][1]
                third_level_precondition_implausible_story_attn = attn_mean_physical_states[1][1]
            else:
                third_level_effect_implausible_story_attn = attn_mean_physical_states[0][0]
                third_level_precondition_implausible_story_attn = attn_mean_physical_states[1][0]
            third_level_conflict_sentences_attn_mean = (third_level_effect_implausible_story_attn[confl_sentence_index_1] + third_level_precondition_implausible_story_attn[confl_sentence_index_2]) / 2
            third_level_irrelevant_sentences_attn_mean = (np.mean(np.delete(third_level_effect_implausible_story_attn, confl_sentence_index_1)) + np.mean(np.delete(third_level_precondition_implausible_story_attn, confl_sentence_index_2))) / 2
            num_correct_sentences += 1
            if third_level_conflict_sentences_attn_mean > sentences_attn_threshold:
                num_sentences_attn_exceed_threshold += 1
                if correct_physical_states:
                    num_true_positive_third_level += 1
                else:
                    num_false_positive_third_level += 1
            else:
                if correct_physical_states:
                    num_false_negative_third_level += 1
                else:
                    num_true_negative_third_level += 1
            correct_sentences_total_attn += third_level_conflict_sentences_attn_mean
            incorrect_sentences_total_attn += third_level_irrelevant_sentences_attn_mean
        
        story_A_attention_second_level_scaled = (np.mean(attn_mean_conflict[0][0]) / (np.mean(attn_mean_conflict[0][0]) + np.mean(attn_mean_conflict[0][1]))) * 100
        story_B_attention_second_level_scaled = (np.mean(attn_mean_conflict[0][1]) / (np.mean(attn_mean_conflict[0][0]) + np.mean(attn_mean_conflict[0][1]))) * 100

        if correct_confl_pairs:
            print("second level correct")
            sentences_attention_third_level = np.concatenate((np.mean(np.stack((attn_mean_physical_states[0][0], attn_mean_physical_states[1][0])), axis=0), np.mean(np.stack((attn_mean_physical_states[0][1], attn_mean_physical_states[1][1])), axis=0)))
            sentences_attention_third_level_scaled = sentences_attention_third_level * 100
        else:
            print("second level incorrect")
            sentences_attention_third_level_scaled = np.zeros(len(stories_info[0]['sentences']) + len(stories_info[1]['sentences']))

        story_pair_lines = story_prompt.split('\n')
        if story_pair_lines[-1] == '':
            story_pair_lines = story_pair_lines[:-1]
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
        for i in range(1, len(stories_info[0]['sentences']) + 1):
            for char in story_pair_lines[i]:
                context_string += char
                attention_list_second_level.append(story_A_attention_second_level_scaled)
                attention_list_third_level.append(sentences_attention_third_level_scaled[i - 1])
            context_string += '\n'
            attention_list_second_level.append(0) # account \n
            attention_list_third_level.append(0) # account \n
        for char in story_pair_lines[len(stories_info[0]['sentences']) + 1]:
            context_string += char
            attention_list_second_level.append(0) # account for story B identifier
            attention_list_third_level.append(0) # account for story B identifier
        context_string += '\n'
        attention_list_second_level.append(0) # account \n
        attention_list_third_level.append(0) # account \n
        for i in range(2 + len(stories_info[0]['sentences']), 2 + len(stories_info[0]['sentences']) + len(stories_info[1]['sentences'])):
            for char in story_pair_lines[i]:
                context_string += char
                attention_list_second_level.append(story_B_attention_second_level_scaled)
                attention_list_third_level.append(sentences_attention_third_level_scaled[i - 2])
            context_string += '\n'
            attention_list_second_level.append(0) # account \n
            attention_list_third_level.append(0) # account \n

        if not os.path.exists("attention_visualization_results"):
            os.makedirs("attention_visualization_results")
          
        # generate_heatmap_upon_text(context_string, attention_list_second_level, "attention_visualization_results/trip_baseline_second_level_attn_vis.tex", color='red')
        generate_heatmap_upon_text(context_string, attention_list_third_level, "attention_visualization_results/trip_baseline_third_level_attn_vis.tex", color='red')
