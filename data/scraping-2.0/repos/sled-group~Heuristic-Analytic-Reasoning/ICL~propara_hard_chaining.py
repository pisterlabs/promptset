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
import json
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from data.propara import load_propara_dataset, plausibility_demo_full, conflict_demo_full, physical_states_demo_full, story_pair_prompt_generator, story_prompt_generator
from eval.propara import check_response, response_extractor
from models import get_chat_message, API_COSTS, prompt_gpt3_with_caching, prompt_chat_gpt_with_caching, VICUNA13B_PATH, prompt_llama_with_caching
from utils import get_output_dir_name

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lm_backbone", default="llama7b", choices=["gpt3", "gpt4", "chatgpt", 'llama7b', 'llama13b', 'llama30b', 'llama65b'])
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--local_only", action="store_true", default=False)
    parser.add_argument("--demo_choice", default="stories-4", choices=["stories-4", "balanced-6"])
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--cache_only", action="store_true", default=False)

    parser.add_argument("--action_type", type=str, default="conversion", choices=["conversion"])

    parser.add_argument("--output_dir", type=str, default="saved_results")
    parser.add_argument("--debug", action='store_true', help="Whether to run the model on only a small amount of data.")
    parser.add_argument("--skip_prompting", action='store_true', help="Whether to NOT prompt any LMs, just generate prompts and inspect them.")
    args = parser.parse_args()

    LETTER_TO_IDX = {'A': 0, 'B': 1}

    LM_BACKBONE_ENGINES = {'gpt4': 'gpt-4', 'chatgpt': 'gpt-35-turbo', 'alpaca13b': 'alpaca13b'}

    action_types = [args.action_type]
    dataset = load_propara_dataset(action_types=action_types)
    train_dataset = dataset[args.action_type]["train"]
    test_dataset = dataset[args.action_type]["test"]
    if args.debug:
        test_dataset = test_dataset[:5]

    CACHE_FNAME = f'cache_files/propara_lm_cache_{args.lm_backbone}_{args.demo_choice}_hard-chaining.pkl'
    print("Using CACHE_FNAME:", CACHE_FNAME)
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

    if args.demo_choice == "stories-4":
        selected_story_example_ids = [200, 68, 107, 311]
    # elif args.demo_choice == "balanced-6":
    #     # sample 2 stories for 3 types of conflict
    #     selected_story_example_ids = balanced_sample_story_ids(train_dataset, select_n_stories_for_each_category=2) # If encounters any error, change a random seed!
    else:
        raise Exception("demo_choice out of options")
    
    print("selected story IDs:", selected_story_example_ids)
    
    # selected_story_conflict_reasons = {
    #     588: "the donut was put in the trash in sentence 4, but Mary ate the donut in sentence 5",
    #     107: "in sentence 1 Ann realizes that all of her tools are missing yet in sentence 2 she is taking out an axe, a pair of scissors, rope and a few other things",
    #     0: "Tom put the soup in the microwave and then ate it cold",
    #     51: "Sentence 3 states that the hair dryer would not turn on, while Sentence 4 states that Ann used the hair dryer to dry her hair",
    #     311: "you cannot turn something on that is not plugged in",
    # }

    # Generate some demos for story plausibility
    plausibility_demos = ""
    for selected_story_id in selected_story_example_ids:
        story_pair = train_dataset[selected_story_id]
        plausibility_demos += plausibility_demo_full(story_pair) + "\n"

    # Generate some demos for conflict detection
    conflict_demos = ""
    for selected_story_id in selected_story_example_ids:
        story_pair = train_dataset[selected_story_id]
        conflict_demos += conflict_demo_full(story_pair, mode="no_story") + "\n"

    # # Demos for action precondition prediction (APP) and action effect prediction (AEP)
    # app_demos = generate_app_demos(train_dataset)
    # aep_demos = generate_aep_demos(train_dataset)

    # Generate some demos for physical states
    # physical_states_demos = app_demos + "\n" + aep_demos
    physical_states_demos = ""
    for selected_story_id in selected_story_example_ids:
        story_pair = train_dataset[selected_story_id]
        physical_states_demos += physical_states_demo_full(story_pair, mode="sentence") + "\n"
        

    # Start testing
    predictions = []
    plausibility_score, conflict_score, physical_states_score = 0, 0, 0
    
    for i in tqdm(range(0, len(test_dataset))):

        example_id = str(test_dataset[i]['example_id'])

        prediction_obj = {'example_id': example_id}
        story_pair = test_dataset[i]
        if story_pair['story_converted'] == 'A':
            story_converted = story_pair["story_A_sentences"]
        else:
            story_converted = story_pair["story_B_sentences"]
        story_prompt = story_pair_prompt_generator(story_pair)

        # STEP 1: Prompt LM for which story is plausible
        if args.lm_backbone in ['gpt3']:
            plausibility_prompt = plausibility_demos
            plausibility_prompt += "\n" + story_pair_prompt_generator(story_pair)
            prediction_obj['input_prompt_plausibility'] = plausibility_prompt.split('\n')
            plausibility_generated_text = prompt_gpt3_with_caching(plausibility_prompt, args, lm_cache, example_id + '_plausibility', token_counts=TOKEN_COUNTS)
        elif args.lm_backbone in ['gpt4', 'chatgpt']:
            plausibility_prompt = []
            plausibility_prompt.append(get_chat_message('system', preamble_plausibility))
            plausibility_prompt.append(get_chat_message('system', "Here are some examples:\n\n" + plausibility_demos))
            plausibility_prompt.append(get_chat_message('user', story_pair_prompt_generator(story_pair)))
            prediction_obj['input_prompt_plausibility'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in plausibility_prompt]
            plausibility_generated_text = prompt_chat_gpt_with_caching(plausibility_prompt, args, lm_cache, example_id + '_plausibility', token_counts=TOKEN_COUNTS)
        elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
            plausibility_prompt = plausibility_demos
            plausibility_prompt += "\n" + story_pair_prompt_generator(story_pair)
            plausibility_generated_text = prompt_llama_with_caching(model, tokenizer, plausibility_prompt, args, lm_cache, example_id + '_plausibility', max_tokens=128)
        if args.lm_backbone in ['gpt3', 'gpt4', 'chatgpt']:
            response = response_extractor(plausibility_generated_text, type='plausibility')
            predicted_plausible_story_letter = response['plausibility']['story_converted']
        elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
            response = response_extractor(plausibility_generated_text, type='plausibility')
            predicted_plausible_story_letter = response['plausibility']['story_converted']
        prediction_obj['plausible_story'] = predicted_plausible_story_letter
        prediction_obj['generated_text_plausibility'] = plausibility_generated_text.split('\n')

        # If LM was correct, separately ask for the evidence for the end task prediction
        prediction_obj['confl_pairs'] = []
        prediction_obj['physical_states'] = []
        if check_response(plausibility_generated_text, story_pair, demo_type="plausibility"):
            # Picked the correct story!
            plausibility_score += 1
            story_name = f"story_{predicted_plausible_story_letter}_sentences"

            # STEP 2: Prompt LM for which sentences are conflicting
            if args.lm_backbone in ['gpt3']:
                conflict_prompt = conflict_demos
                conflict_prompt += "\n" + story_prompt_generator(story_pair, story_name)
                prediction_obj['input_prompt_conflict'] = conflict_prompt.split('\n')
                conflict_generated_text = prompt_gpt3_with_caching(conflict_prompt, args, lm_cache, example_id + '_conflict', token_counts=TOKEN_COUNTS)
            elif args.lm_backbone in ['gpt4', 'chatgpt']:
                conflict_prompt = []
                conflict_prompt.append(get_chat_message("system", preamble_conflict))
                conflict_prompt.append(get_chat_message("system", "Here are some examples:\n\n" + conflict_demos))
                conflict_prompt.append(get_chat_message("user", story_prompt_generator(story_pair, story_name)))
                prediction_obj['input_prompt_conflict'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in conflict_prompt]
                conflict_generated_text = prompt_chat_gpt_with_caching(conflict_prompt, args, lm_cache, example_id + '_conflict', token_counts=TOKEN_COUNTS)
            elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
                conflict_prompt = conflict_demos
                conflict_prompt += "\n" + story_prompt_generator(story_pair, story_name)
                conflict_generated_text = prompt_llama_with_caching(model, tokenizer, conflict_prompt, args, lm_cache, example_id + '_conflict', max_tokens=128)
            response = response_extractor(conflict_generated_text, type='conflict_short')
            prediction_obj['conv_sentence'] = response['conflict']['state_converted_to']
            prediction_obj['generated_text_conflict'] = conflict_generated_text

            if check_response(conflict_generated_text, story_pair, demo_type="conflict_short"):
                # Picked the correct conflicting sentences!
                conflict_score += 1
                story = story_pair['story_converted'].capitalize()
                sentence = story_pair['conversions'][0]['state_converted_to']
                sentence_prompt = story_pair[f"story_{story}_sentences"][sentence - 1] + " "
                sentence_prompt += f"What happened to {story_pair['participant_converted']}?\n"

                # STEP 3: Prompt LM for physical states in conflicting sentences
                if args.lm_backbone in ['gpt3']:
                    physical_states_prompt = physical_states_demos
                    physical_states_prompt += "\n" + sentence_prompt
                    prediction_obj['input_prompt_physical_states'] = physical_states_prompt.split('\n')
                    physical_states_generated_text = prompt_gpt3_with_caching(physical_states_prompt, args, lm_cache, example_id + '_physical_states', token_counts=TOKEN_COUNTS)
                elif args.lm_backbone in ['gpt4', 'chatgpt']:
                    physical_states_prompt = []
                    physical_states_prompt.append(get_chat_message("system", preamble_physical_states))
                    physical_states_prompt.append(get_chat_message("system", "Here are some examples:\n\n" + physical_states_demos))
                    physical_states_prompt.append(get_chat_message("user", sentence_prompt))
                    prediction_obj['input_prompt_aep'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in physical_states_prompt]
                    physical_states_generated_text = prompt_chat_gpt_with_caching(physical_states_prompt, args, lm_cache, example_id + '_physical_states', token_counts=TOKEN_COUNTS)    
                elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
                    physical_states_prompt = physical_states_demos
                    physical_states_prompt += "\n" + sentence_prompt
                    physical_states_generated_text = prompt_llama_with_caching(model, tokenizer, physical_states_prompt, args, lm_cache, example_id + '_physical_states', max_tokens=128)  
                response = response_extractor(physical_states_generated_text, type='physical_states')
                prediction_obj['participant_converted_to'] = response['physical_states']['participant_converted_to']
                prediction_obj['generated_text_physical_states'] = physical_states_generated_text

                if check_response(physical_states_generated_text, story_pair, demo_type="physical_states"):
                    # Picked the correct physical states!
                    physical_states_score += 1
                    print("Correct!")

        predictions.append(prediction_obj)

    plausibility_score /= len(test_dataset)
    conflict_score /= len(test_dataset)
    physical_states_score /= len(test_dataset)
    if not args.skip_prompting:
        print("Accuracy_full:", plausibility_score)
        print("Consistency_full:", conflict_score)
        print("Verifiability_full:", physical_states_score)
        metrics = {
            'accuracy': plausibility_score,
            'consistency': conflict_score,
            'verifiability': physical_states_score
        }

        # Create output directory for results
        instance_timestamp = datetime.datetime.now()
        output_dir, instance_name = get_output_dir_name('propara', args, instance_timestamp, 0, result_type="hard-chaining")
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

