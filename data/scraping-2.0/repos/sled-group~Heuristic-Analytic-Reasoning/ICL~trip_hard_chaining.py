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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from data.trip import load_trip_dataset
from eval.trip import trip_metrics, \
                      story_pair_demo_generator_topdown, story_pair_demo_generator_topdown_ask_implausible, story_pair_prompt_generator, story_prompt_generator, \
                      conflict_demo_generator, conflict_prompt_generator, confl_pairs_extractor_topdown, \
                      generate_aep_demos, generate_app_demos, app_prompt_generator, aep_prompt_generator, extract_and_combine_physical_states_topdown, \
                      implausible_story_conflict_reason_generator, balanced_sample_story_ids
from models import prompt_gpt3_with_caching, prompt_chat_gpt_with_caching, get_chat_message, API_COSTS, prompt_llama_with_caching
from utils import get_output_dir_name

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lm_backbone", default="gpt3", choices=["gpt3", "gpt4", "chatgpt", "llama7b", "llama13b", "llama30b", "llama65b"])
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--local_only", action="store_true", default=False)
    parser.add_argument("--demo_choice", default="stories-4", choices=["stories-4", "balanced-6"])
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--ask_implausible", type=bool, default=False)

    parser.add_argument("--use_conflict_explanations", action="store_true")
    parser.add_argument("--exclude_multi_confl", default=False, action="store_true")
    parser.add_argument("--condense_multiple_conflict", default=False, action="store_true")
    parser.add_argument("--cache_only", default=False, action="store_true")
    parser.add_argument("--reduce_options", default=False, action="store_true")
    parser.add_argument("--output_attn", action="store_true", default=False)
    
    parser.add_argument("--output_dir", type=str, default="saved_results")
    parser.add_argument("--debug", action='store_true', help="Whether to run the model on only a small amount of data.")
    parser.add_argument("--skip_prompting", action='store_true', help="Whether to NOT prompt any LMs, just generate prompts and inspect them.")
    args = parser.parse_args()

    LM_BACKBONE_ENGINES = {'gpt4': 'gpt-4', 'chatgpt': 'gpt-35-turbo'}
    LETTER_TO_IDX = {'A': 0, 'B': 1}

    train_dataset = load_trip_dataset(exclude_multi_confl=False, condense_multiple_conflict=False, reduce_options=False)["train"]
    test_dataset = load_trip_dataset(exclude_multi_confl=args.exclude_multi_confl, condense_multiple_conflict=args.condense_multiple_conflict, reduce_options=args.reduce_options)["test"]
    if args.debug:
        test_dataset = test_dataset[:5]

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

    CACHE_FNAME = f'cache_files/trip_lm_cache_{args.lm_backbone}_{args.demo_choice}_hard-chaining.pkl'
    print("Using CACHE_FNAME:", CACHE_FNAME)
    if args.use_conflict_explanations:
        CACHE_FNAME = CACHE_FNAME.replace('.pkl', '_expl.pkl')
    if os.path.exists(CACHE_FNAME):
        lm_cache = pickle.load(open(CACHE_FNAME, 'rb'))
        print(len(lm_cache))
    else:
        lm_cache = {}
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
        preamble_app = preamble + (
            "report the physical *effect* states of entities in each story by answering questions about them before different actions are taken in the story. "
            "The story may not explicitly mention state changes. In these cases, you have to use commonsense reasoning to infer the state changes. "
            "However, you will be penalized if you incorrectly predict an entity's physical state, so you should only predict states you're absolutely sure of based on the text. "
        )
        preamble_aep = preamble + (
            "report the physical *precondition* states of entities in each story by answering questions about them after different actions are taken in the story. "
            "The story may not explicitly mention state changes. In these cases, you have to use commonsense reasoning to infer the state changes. "
            "However, you will be penalized if you incorrectly predict an entity's physical state, so you should only predict states you're absolutely sure of based on the text. "
        )

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
        if args.ask_implausible:
            plausibility_demos += story_pair_demo_generator_topdown_ask_implausible(story_pair, 
                                                                conflict_reason_enabled=True if args.use_conflict_explanations else False,
                                                                reasoning_depth="accurate") + "\n"
        else:
            plausibility_demos += story_pair_demo_generator_topdown(story_pair, 
                                                                conflict_reason_enabled=True if args.use_conflict_explanations else False,
                                                                reasoning_depth="accurate") + "\n"

    # Generate some demos for conflict detection
    conflict_demos = ""
    for sampled_train_data_i, selected_story_id in enumerate(selected_story_example_ids):
        demo_example = [example for example in train_dataset if example['example_id'] == selected_story_id][0]
        story_pair = demo_example['stories']
        implausible_story = story_pair[0] if not story_pair[0]['plausible'] else story_pair[1]
        sentence_numbers = (implausible_story['confl_pairs'][0][0] + 1, implausible_story['confl_pairs'][0][1] + 1)
        conflict_demos += story_prompt_generator(implausible_story)
        conflict_demo_temp = conflict_demo_generator(sentence_numbers,
                                                     conflict_reason= implausible_story_conflict_reason_generator(implausible_story) if args.use_conflict_explanations else None)
        conflict_demo_temp = conflict_demo_temp if args.use_conflict_explanations else conflict_demo_temp.capitalize()
        conflict_demos += conflict_demo_temp + "\n"

    # Demos for action precondition prediction (APP) and action effect prediction (AEP)
    app_demos = generate_app_demos(train_dataset, reduce_options=args.reduce_options)
    aep_demos = generate_aep_demos(train_dataset, reduce_options=args.reduce_options)

    # print(plausibility_demos)
    # print(conflict_demos)
    # print(app_demos)
    # print(aep_demos)

    predictions = []
    for i in tqdm(range(0, len(test_dataset))):

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
            plausibility_generated_text = prompt_gpt3_with_caching(plausibility_prompt, args, lm_cache, example_id + '_plausibility', token_counts=TOKEN_COUNTS)
        elif args.lm_backbone in ["llama7b", "llama13b", "llama30b", "llama65b"]:
            plausibility_prompt = plausibility_demos
            plausibility_prompt += "\n" + story_pair_prompt_generator(stories_info[0], stories_info[1])
            prediction_obj['input_prompt_plausibility'] = plausibility_prompt.split('\n')
            plausibility_generated_text= prompt_llama_with_caching(model, tokenizer, plausibility_prompt, args, lm_cache, example_id + '_plausibility')
        elif args.lm_backbone in ['gpt4', 'chatgpt']:
            plausibility_prompt = []
            plausibility_prompt.append(get_chat_message('system', preamble_plausibility))
            plausibility_prompt.append(get_chat_message('system', "Here are some examples:\n\n" + plausibility_demos))
            plausibility_prompt.append(get_chat_message('user', story_pair_prompt_generator(stories_info[0], stories_info[1])))
            prediction_obj['input_prompt_plausibility'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in plausibility_prompt]
            plausibility_generated_text = prompt_chat_gpt_with_caching(plausibility_prompt, args, lm_cache, example_id + '_plausibility', token_counts=TOKEN_COUNTS)
        if args.ask_implausible:
            predicted_plausible_story_letter = None
            predicted_implausible_story_letter = plausibility_generated_text.split(' ')[1][:1]
            if predicted_implausible_story_letter == 'A':
                predicted_plausible_story_letter == 'B'
            else:
                predicted_plausible_story_letter == 'A'
        else:
            predicted_plausible_story_letter = plausibility_generated_text.split(' ')[1][:1] # Just take second token as the plausible story
        predicted_implausible_story = stories_info[1 - LETTER_TO_IDX[predicted_plausible_story_letter]]
        prediction_obj['plausible_story'] = predicted_plausible_story_letter
        prediction_obj['generated_text_plausibility'] = plausibility_generated_text.split('\n')

        # If LM was correct, separately ask for the evidence for the end task prediction
        prediction_obj['confl_pairs'] = []
        prediction_obj['physical_states'] = []
        if stories_info[LETTER_TO_IDX[predicted_plausible_story_letter]]['plausible']:
            # Picked the correct story!

            # STEP 2: Prompt LM for which sentences are conflicting
            if args.lm_backbone in ['gpt3']:
                conflict_prompt = conflict_demos
                conflict_prompt += story_prompt_generator(predicted_implausible_story)
                conflict_prompt += conflict_prompt_generator()
                prediction_obj['input_prompt_conflict'] = conflict_prompt.split('\n')
                conflict_generated_text = prompt_gpt3_with_caching(conflict_prompt, args, lm_cache, example_id + '_conflict', token_counts=TOKEN_COUNTS)
            elif args.lm_backbone in ["llama7b", "llama13b", "llama30b", "llama65b"]:
                conflict_prompt = conflict_demos
                conflict_prompt += story_prompt_generator(predicted_implausible_story)
                conflict_prompt += conflict_prompt_generator()
                prediction_obj['input_prompt_conflict'] = conflict_prompt.split('\n')
                conflict_generated_text = prompt_llama_with_caching(model, tokenizer, conflict_prompt, args, lm_cache, example_id + '_conflict')
            elif args.lm_backbone in ['gpt4', 'chatgpt']:
                conflict_prompt = []
                conflict_prompt.append(get_chat_message("system", preamble_conflict))
                conflict_prompt.append(get_chat_message("system", "Here are some examples:\n\n" + conflict_demos))
                conflict_prompt.append(get_chat_message("user", story_prompt_generator(predicted_implausible_story) + conflict_prompt_generator()))
                prediction_obj['input_prompt_conflict'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in conflict_prompt]
                conflict_generated_text = prompt_chat_gpt_with_caching(conflict_prompt, args, lm_cache, example_id + '_conflict', token_counts=TOKEN_COUNTS)
            predicted_confl_pair = confl_pairs_extractor_topdown(conflict_generated_text.split(' '))
            prediction_obj['confl_pairs'] = predicted_confl_pair
            prediction_obj['generated_text_conflict'] = conflict_generated_text

            if predicted_confl_pair == implausible_story['confl_pairs']:
                # Picked the correct conflicting sentences!

                # STEP 3: Prompt LM for physical states in conflicting sentences
                if args.lm_backbone in ['gpt3']:
                    # First prompt for effect states of first conflicting sentence
                    aep_prompt = aep_demos + "\n"
                    aep_prompt += aep_prompt_generator(predicted_implausible_story['sentences'][min(predicted_confl_pair[0])][:-1])
                    prediction_obj['input_prompt_aep'] = aep_prompt.split('\n')
                    aep_generated_text = prompt_gpt3_with_caching(aep_prompt, args, lm_cache, example_id + '_aep', token_counts=TOKEN_COUNTS)

                    # Then prompt for effect states of second conflicting sentence
                    app_prompt = app_demos + "\n"
                    app_prompt += app_prompt_generator(predicted_implausible_story['sentences'][max(predicted_confl_pair[0])][:-1])
                    prediction_obj['input_prompt_app'] = app_prompt.split('\n')
                    app_generated_text = prompt_gpt3_with_caching(app_prompt, args, lm_cache, example_id + '_app', token_counts=TOKEN_COUNTS)
                elif args.lm_backbone in ["llama7b", "llama13b", "llama30b", "llama65b"]:
                    # First prompt for effect states of first conflicting sentence
                    aep_prompt = aep_demos + "\n"
                    aep_prompt += aep_prompt_generator(predicted_implausible_story['sentences'][min(predicted_confl_pair[0])][:-1])
                    prediction_obj['input_prompt_aep'] = aep_prompt.split('\n')
                    aep_generated_text = prompt_llama_with_caching(model, tokenizer, aep_prompt, args, lm_cache, example_id + '_aep')

                    # Then prompt for effect states of second conflicting sentence
                    app_prompt = app_demos + "\n"
                    app_prompt += app_prompt_generator(predicted_implausible_story['sentences'][max(predicted_confl_pair[0])][:-1])
                    prediction_obj['input_prompt_app'] = app_prompt.split('\n')
                    app_generated_text = prompt_llama_with_caching(model, tokenizer, app_prompt, args, lm_cache, example_id + '_app')
                elif args.lm_backbone in ['gpt4', 'chatgpt']:
                    # First prompt for effect states of first conflicting sentence
                    aep_prompt = []
                    aep_prompt.append(get_chat_message("system", preamble_aep))
                    aep_prompt.append(get_chat_message("system", "Here are some examples:\n\n" + aep_demos))
                    aep_prompt.append(get_chat_message("user", aep_prompt_generator(predicted_implausible_story['sentences'][min(predicted_confl_pair[0])][:-1])))
                    prediction_obj['input_prompt_aep'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in aep_prompt]
                    aep_generated_text = prompt_chat_gpt_with_caching(aep_prompt, args, lm_cache, example_id + '_aep', token_counts=TOKEN_COUNTS)
  
                    # Then prompt for effect states of second conflicting sentence
                    app_prompt = []
                    app_prompt.append(get_chat_message("system", preamble_app))
                    app_prompt.append(get_chat_message("system", "Here are some examples:\n\n" + app_demos))
                    app_prompt.append(get_chat_message("user", app_prompt_generator(predicted_implausible_story['sentences'][max(predicted_confl_pair[0])][:-1])))
                    prediction_obj['input_prompt_app'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in app_prompt]
                    app_generated_text = prompt_chat_gpt_with_caching(app_prompt, args, lm_cache, example_id + '_app', token_count=TOKEN_COUNTS)  

                prediction_obj['physical_states'] = extract_and_combine_physical_states_topdown("what is the state of " + aep_generated_text + '\n\nwhat was the state of ' + app_generated_text,
                                                                                                prediction_obj, 
                                                                                                implausible_story['entities'],
                                                                                                implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][0]],
                                                                                                implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][1]],)
                prediction_obj['generated_text_aep'] = aep_generated_text
                prediction_obj['generated_text_app'] = app_generated_text

        predictions.append(prediction_obj)        

    if not args.skip_prompting:
        metrics, predictions = trip_metrics(test_dataset, predictions)
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
        output_dir, instance_name = get_output_dir_name('trip', args, instance_timestamp, 0, result_type="hard-chaining")
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


