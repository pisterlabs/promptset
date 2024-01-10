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
                      story_pair_demo_generator_topdown, story_pair_demo_generator_topdown_ask_implausible, story_pair_prompt_generator, conflict_demo_generator_fully_sep, \
                      plausibility_extractor_cot, \
                      confl_pairs_extractor_topdown, physical_states_demo_generator_fully_separate, \
                      generate_aep_demos_fully_separate_familiarization, generate_app_demos_fully_separate_familiarization, extract_and_combine_physical_states_cot, \
                      implausible_story_conflict_reason_generator, balanced_sample_story_ids
from models import get_chat_message, API_COSTS, prompt_gpt3_with_caching, prompt_chat_gpt_with_caching, VICUNA13B_PATH, prompt_llama_with_caching, get_65b_device_mapping
from utils import get_output_dir_name
from visualization import separation_generator

import pandas as pd

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lm_backbone", default="gpt3", choices=["gpt3", "gpt4", "chatgpt", "vicuna", "llama7b", "llama13b", "llama30b", "llama65b"])
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--local_only", action="store_true", default=False)
    parser.add_argument("--demo_choice", default="stories-4", choices=["stories-4", "balanced-6"])
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--num_demos", type=int, default=0)
    parser.add_argument("--ask_implausible", type=bool, default=False)

    parser.add_argument("--use_conflict_explanations", action="store_true")
    parser.add_argument("--exclude_multi_confl", default=False, action="store_true")
    parser.add_argument("--condense_multiple_conflict", default=False, action="store_true")
    parser.add_argument("--cache_only", action="store_true", default=False)
    parser.add_argument("--reduce_options", action="store_true", default=False)
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

    CACHE_FNAME = f'cache_files/trip_lm_cache_{args.lm_backbone}_{args.demo_choice}_baseline-separate-cot.pkl'
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
        story_A = story_pair[0]
        story_B = story_pair[1]
        # if args.ask_implausible:
        #     plausibility_demos += story_pair_demo_generator_topdown_ask_implausible(story_pair, 
        #                                                         conflict_reason_enabled=True if args.use_conflict_explanations else False,
        #                                                         reasoning_depth="accurate") + "\n"
        # else:
        #     plausibility_demos += story_pair_demo_generator_topdown(story_pair, 
        #                                                         conflict_reason_enabled=True if args.use_conflict_explanations else False,
        #                                                         reasoning_depth="accurate") + "\n"
        # print(plausibility_demos)
        if selected_story_id == "train_588":
            plausibility_demos += story_pair_prompt_generator(story_A, story_B)
            plausibility_demos += "Let’s think step by step about which story is more plausible. In Story A, Mary takes a cucumber and a donut out of the fridge, and then throws away the donut. This is not a very effective way to eat a donut, so it’s not very plausible. In Story B, Mary takes a cucumber and a donut out of the fridge, and then eats the donut. Therefore, Story B is more plausible. \n"
        if selected_story_id == "train_107":
            plausibility_demos += story_pair_prompt_generator(story_A, story_B)
            plausibility_demos += "Let’s think step by step about which story is more plausible. In Story A, the fact that Ann takes an axe, scissors, and rope might make us think that she’s planning to do something nefarious. In Story B, it might simply be that she’s going camping and needs these items. The two stories both start out the same, but Story A quickly diverges into something much less likely. Therefore, Story B is more plausible. \n"
        if selected_story_id == "train_51":
            plausibility_demos += story_pair_prompt_generator(story_A, story_B)
            plausibility_demos += "Let’s think step by step about which story is more plausible. In story A, after Ann tries to use the hair dryer, we might think that her shorts getting wet in the shower is a direct consequence of the hair dryer not working. In story B, on the other hand, it is much more plausible that Ann’s shorts got wet in the shower because that is what usually happens when you take a shower. Therefore, Story B is more plausible. \n"
        if selected_story_id == "train_311":
            plausibility_demos += story_pair_prompt_generator(story_A, story_B)
            plausibility_demos += "Let’s think step by step about which story is more plausible. In story A, John is looking for a notebook, and he finds it on the coffee table. This is plausible. In story B, John is looking for a notebook, but he turns on the TV instead. This is not as plausible, because it’s not likely that John would forget what he was looking for and turn on the TV. Therefore, Story A is more plausible. \n"
        # example_gpt_plausibility_prompt = story_pair_prompt_generator(story_pair[0], story_pair[1])
        # example_gpt_plausibility_prompt += "Which story is more plausible? Let's think step by step."
        # print(example_gpt_plausibility_prompt)
        # plausibility_generated_text = prompt_gpt3_with_caching(example_gpt_plausibility_prompt, args, lm_cache, selected_story_id + '_plausibility', token_counts=TOKEN_COUNTS)
        # print(repr(plausibility_generated_text))

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
        # conflict_demos += story_pair_prompt_generator(story_A, story_B)
        # sentence_numbers = (implausible_story['confl_pairs'][0][0] + 1, implausible_story['confl_pairs'][0][1] + 1)
        # conflict_demo_temp = conflict_demo_generator_fully_sep(sentence_numbers, implausible_story_letter,
        #                                              conflict_reason=implausible_story_conflict_reason_generator(implausible_story) if args.use_conflict_explanations else None)
        # print(conflict_demo_temp)
        # conflict_demo_temp = conflict_demo_temp if args.use_conflict_explanations else conflict_demo_temp.capitalize()
        # conflict_demos += conflict_demo_temp + "\n"
        if selected_story_id == "train_588":
            conflict_demos += story_pair_prompt_generator(story_A, story_B)
            conflict_demos += "Let’s think step by step about which sentences are conflicting in one story. In Story A, Mary took the bowl out of the fridge, saw that it had a cucumber and a donut, put the cucumber on a plate, and then threw the donut away. In Story B, Mary also took the bowl out of the fridge and saw that it had a cucumber and a donut, but instead of throwing the donut away, she ate it. Therefore, sentences 4 and 5 conflict with each other in story A. \n"
        if selected_story_id == "train_107":
            conflict_demos += story_pair_prompt_generator(story_A, story_B)
            conflict_demos += "Let’s think step by step about which sentences are conflicting in one story. In story A, Ann opens the toolbox and realizes that her tools are gone. In story B, Ann opens the toolbox to get a few things for her camping trip. These two sentences conflict with each other because in story A, Ann's reaction to opening the toolbox is different than in story B. In story A, Ann is surprised and upset that her tools are missing, while in story B, Ann is simply getting what she needs for her trip. Other sentences that conflict are sentence 3 in story A and sentence 5 in story B. In story A, Ann puts the items she took from the toolbox in a box and then in the trunk of her car. However, in story B, Ann puts the items directly in the trunk of her car. This discrepancy could be due to Ann forgetting to put the items in a box in story B, or it could be intentional because she is in a hurry. Sentence 4 in story A conflicts with sentence 5 in story B because in story A, Ann fastens her bicycle to the bicycle rack before going inside to get the rest of her gear. However, in story B, Ann goes inside to get the rest of her gear before fastening her bicycle to the bicycle rack. This discrepancy could be due to Ann forgetting to fasten her bicycle in story B, or it could be intentional because she is in a hurry. Therefore, sentences 1 and 2 conflict with each other in story A. \n"
        if selected_story_id == "train_51":
            conflict_demos += story_pair_prompt_generator(story_A, story_B)
            conflict_demos += "Let’s think step by step about which sentences are conflicting in one story. In story A, Ann tries to use a hair dryer but it doesn’t work because the cord is broken. In story B, there is no mention of a broken cord, so we can assume that the hair dryer does work. This is a conflict. Another conflict is that in story A, Ann plugs in the hair dryer before using it, but in story B, there is no mention of this. We can assume that Ann doesn’t plug in the hair dryer in story B, which is another conflict. Therefore, sentences 3 and 4 conflict with each other in story A. \n"
        if selected_story_id == "train_311":
            conflict_demos += story_pair_prompt_generator(story_A, story_B)
            conflict_demos += "Let’s think step by step about which sentences are conflicting in one story. In story A, John searches for the notebook and then finds it. However, in story B, John searches for the notebook but then turns on the TV. The actions are in conflict because John cannot do both activities at the same time—he either finds the notebook or he turns on the TV. Therefore, sentences 2 and 5 conflict with each other in story B. \n"
        # example_gpt_conflict_prompt = story_pair_prompt_generator(story_pair[0], story_pair[1])
        # example_gpt_conflict_prompt += "Which two sentences are conflicting with each other? Let's think step by step."
        # print(example_gpt_conflict_prompt)
        # conflict_generated_text = prompt_gpt3_with_caching(example_gpt_conflict_prompt, args, lm_cache, selected_story_id + '_conflict', token_counts=TOKEN_COUNTS)
        # print(repr(conflict_generated_text))

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
        # physical_states_demos += "\n" + story_pair_prompt_generator(story_A, story_B)
        # print(physical_states_demo_generator_fully_separate(implausible_story))
        # physical_states_demos += physical_states_demo_generator_fully_separate(implausible_story)

        if selected_story_id == "train_588":
            physical_states_demos += story_pair_prompt_generator(story_A, story_B)
            physical_states_demos += "Let’s think step by step about which physical states are conflicting in two sentences in one story. The final states in the story are not the same, but the difference is the cucumber. In one story, the cucumber is on a plate, while in the other story, the cucumber remains on the counter. We can see that there is a conflict between the physical states of the cucumber in Story A and Story B. In Story A, Mary put the cucumber on a plate, while in Story B, Mary left the cucumber on the counter. Therefore: \nAfter, what is the state of the donut? The donut is now inedible. \nBefore, what was the state of the donut? The donut was edible. \n"
        if selected_story_id == "train_107":
            physical_states_demos += story_pair_prompt_generator(story_A, story_B)
            physical_states_demos += "Let’s think step by step about which physical states are conflicting in two sentences in one story. In Story A, Ann is surprised that her tools are gone when she opens the box. However, in Story B, Ann is expecting to find her camping gear inside the box. The mental state of Ann is different in the two stories, which causes the physical states to be different. Therefore: \nAfter, what are the state of the tools? The tools are now no longer existent. \nBefore, what were the state of the things? The things were existent. \n"
        if selected_story_id == "train_51":
            physical_states_demos += story_pair_prompt_generator(story_A, story_B)
            physical_states_demos += "Let’s think step by step about which physical states are conflicting in two sentences in one story. In other words, let’s find a pair of sentences where the physical state in the first sentence is the opposite of the physical state in the second sentence. 1. Ann used a toothbrush to brush her teeth. 2. Ann walked into the shower. In the first sentence, Ann is using a toothbrush (an object). In the second sentence, Ann is walking (an action). Therefore: \nAfter, what is the state of the hair dryer? The hair dryer is now broken. \nBefore, what was the state of the hair dryer? The hair dryer was functional. \n"
        if selected_story_id == "train_311":
            physical_states_demos += story_pair_prompt_generator(story_A, story_B)
            physical_states_demos += "Let’s think step by step about which physical states are conflicting in two sentences in one story. In Story A, John looks for the notebook and then he finds it. Although we can imagine many different actions that might fulfill these two states, they are in conflict with each other. In Story B, John looks for the notebook and then he turns on the TV. These two states are not in conflict with each other. Therefore: \nAfter, what is the state of the TV? The TV is now unpowered. \nBefore, what was the state of the TV? The TV was powered. \n"

        # example_gpt_state_prompt = story_pair_prompt_generator(story_pair[0], story_pair[1])
        # example_gpt_state_prompt += "What are the conflicting physical states? Let's think step by step."
        # print(example_gpt_state_prompt)
        # state_generated_text = prompt_gpt3_with_caching(example_gpt_state_prompt, args, lm_cache, selected_story_id + '_state', token_counts=TOKEN_COUNTS)
        # print(repr(state_generated_text))

    # print(plausibility_demos)
    # print()
    # print(conflict_demos)
    # print(aep_demos)
    # print(app_demos)
    # print()
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

    for i in tqdm(range(0, trials)):

        acc, cons, ver = 0, 0, 0

        example_id = test_dataset[i]['example_id']

        prediction_obj = {'example_id': example_id}
        stories_info = test_dataset[i]['stories']
        if stories_info[0]['plausible'] == True:
            implausible_story = stories_info[1]
        else:
            implausible_story = stories_info[0]

        # STEP 1: Prompt LM for which story is plausible
        if args.lm_backbone in ['gpt3']:
            plausibility_prompt = plausibility_demos
            plausibility_prompt += "\n" + story_pair_prompt_generator(stories_info[0], stories_info[1])
            prediction_obj['input_prompt_plausibility'] = plausibility_prompt.split('\n')
            # print(plausibility_prompt)
            plausibility_generated_text = prompt_gpt3_with_caching(plausibility_prompt, args, lm_cache, example_id + '_plausibility', max_tokens=256, token_counts=TOKEN_COUNTS)
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
            plausibility_generated_text = prompt_llama_with_caching(model, tokenizer, plausibility_prompt, args, lm_cache, example_id + '_plausibility', max_tokens=256, output_attn=args.output_attn, separations=separations)
        if args.lm_backbone in ['gpt3', 'gpt4', 'chatgpt']:
            predicted_plausible_story_letter = plausibility_extractor_cot(plausibility_generated_text)
        elif args.lm_backbone in ["llama7b", "llama13b", "llama30b", "llama65b"]:
            predicted_plausible_story_letter = plausibility_extractor_cot(plausibility_generated_text) # Just take second token as the plausible story
        # print("predicted_plausible_story_letter:", predicted_plausible_story_letter)
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
                conflict_generated_text = prompt_gpt3_with_caching(conflict_prompt, args, lm_cache, example_id + '_conflict', max_tokens=256, token_counts=TOKEN_COUNTS)
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
                conflict_generated_text = prompt_llama_with_caching(model, tokenizer, conflict_prompt, args, lm_cache, example_id + '_conflict', max_tokens=256, output_attn=args.output_attn, separations=separations)
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
                    physical_states_generated_text = prompt_gpt3_with_caching(physical_states_prompt, args, lm_cache, example_id + '_physical_states', max_tokens=256, token_counts=TOKEN_COUNTS)
                elif args.lm_backbone in ['gpt4', 'chatgpt']:
                    physical_states_prompt = []
                    physical_states_prompt.append(get_chat_message("system", preamble_physical_states))
                    physical_states_prompt.append(get_chat_message("system", "Here are some examples:\n\n"))
                    physical_states_prompt.append(get_chat_message("user", story_pair_prompt_generator(stories_info[0], stories_info[1])))
                    prediction_obj['input_prompt_aep'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in physical_states_prompt]
                    physical_states_generated_text = prompt_chat_gpt_with_caching(physical_states_prompt, args, lm_cache, example_id + '_physical_states', token_counts=TOKEN_COUNTS)    
                elif args.lm_backbone in ["llama7b", "llama13b", "llama30b", "llama65b"]:
                    physical_states_prompt = physical_states_demos
                    physical_states_prompt += "\n" + story_pair_prompt_generator(stories_info[0], stories_info[1])
                    separations = separation_generator(physical_states_demos + "\n", story_pair_prompt_generator(stories_info[0], stories_info[1]), tokenizer) if args.output_attn else None
                    physical_states_generated_text = prompt_llama_with_caching(model, tokenizer, physical_states_prompt, args, lm_cache, example_id + '_physical_states', max_tokens=256, output_attn=args.output_attn, separations=separations)  

                prediction_obj['physical_states'] = extract_and_combine_physical_states_cot(physical_states_generated_text,
                                                                                                prediction_obj, 
                                                                                                implausible_story['entities'],
                                                                                                implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][0]],
                                                                                                implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][1]],)
                prediction_obj['generated_text_physical_states'] = physical_states_generated_text

        predictions.append(prediction_obj)

    metrics, predictions = trip_metrics(test_dataset, predictions)
    if not args.skip_prompting:
        print("Accuracy_full:", metrics['accuracy_full'])
        print("Consistency_full:", metrics['consistency_full'])
        print("Verifiability_full:", metrics['verifiability_full'])
        print("Accuracy_explicit_confl:", metrics['accuracy_explicit_confl'])
        print("Consistency_explicit_confl:", metrics['consistency_explicit_confl'])
        print("Verifiability_explicit_confl:", metrics['verifiability_explicit_confl'])
        print("Accuracy_implicit_confl:", metrics['accuracy_implicit_confl'])
        print("Consistency_implicit_confl:", metrics['consistency_implicit_confl'])
        print("Verifiability_implicit_confl:", metrics['verifiability_implicit_confl'])

        # Create output directory for results
        instance_timestamp = datetime.datetime.now()
        output_dir, instance_name = get_output_dir_name('trip', args, instance_timestamp, 0, result_type="baseline-separate-cot")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        json.dump(predictions, open(os.path.join(output_dir, "preds_test.json"), "w"), indent=4)
        json.dump(metrics, open(os.path.join(output_dir, "metrics_test.json"), "w"), indent=4)
    else:
        if args.lm_backbone in API_COSTS:
            prompt_cost = TOKEN_COUNTS['prompt'] * API_COSTS[args.lm_backbone][0]
            gen_cost = TOKEN_COUNTS['gen'] * API_COSTS[args.lm_backbone][1]
            total_cost = prompt_cost + gen_cost
        else:
            prompt_cost = 0.0
            gen_cost = 0.0
            total_cost = 0.0
        print("PROMPT COST ESTIMATES (EXCLUDING CACHED RESULTS):")
        print("Prompts: %.2f" % prompt_cost)
        print("Generation: %.2f" % gen_cost)
        print("Total: %.2f" % total_cost)