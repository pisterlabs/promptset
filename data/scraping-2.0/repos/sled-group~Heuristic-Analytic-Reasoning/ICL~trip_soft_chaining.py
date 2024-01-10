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
from llama_cpp import Llama
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from fastchat.model import load_model
# from transformers.models.llama.modeling_llama import LlamaModel, LlamaAttention

from data.trip import load_trip_dataset
from eval.trip import trip_metrics, trip_metrics_for_one_pred,\
                      story_pair_demo_generator_topdown, story_pair_demo_generator_topdown_ask_implausible, story_pair_demo_generator_bottomup_compact, story_pair_prompt_generator, \
                      generate_aep_demos, generate_app_demos, generate_aep_demos_fully_separate_familiarization, generate_app_demos_fully_separate_familiarization,\
                      add_trip_preds_topdown, add_trip_preds_topdown_ask_implausible, add_trip_preds_bottomup_compact, \
                      balanced_sample_story_ids
from models import get_chat_message, API_COSTS, prompt_gpt3_with_caching, prompt_llama_with_caching, prompt_chat_gpt_with_caching, prompt_fastchat_with_caching, VICUNA13B_PATH, ALPACA13B_PATH, get_65b_device_mapping
from utils import get_output_dir_name
from visualization import separation_generator

import pandas as pd

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lm_backbone", default="gpt3", choices=["gpt3", "gpt4", "chatgpt", "alpaca13b", "vicuna13b",  "llama7b", "llama13b", "llama30b", "llama65b"])
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--local_only", action="store_true", default=False)
    parser.add_argument("--demo_choice", default="stories-4", choices=["stories-4", "balanced-6"])
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--ask_implausible", type=bool, default=False)
    parser.add_argument("--analyze_attention", action="store_true", default=False)

    parser.add_argument("--reasoning_depth", choices=["accurate", "consistent", "verifiable"], default='verifiable')
    parser.add_argument("--reasoning_direction", choices=["top-down"], default='top-down')
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

    LM_BACKBONE_ENGINES = {'gpt4': 'gpt-4', 'chatgpt': 'gpt-35-turbo', 'alpaca13b': 'alpaca13b'}

    train_dataset = load_trip_dataset(exclude_multi_confl=False, condense_multiple_conflict=False, reduce_options=False)["train"]
    test_dataset = load_trip_dataset(exclude_multi_confl=args.exclude_multi_confl, condense_multiple_conflict=args.condense_multiple_conflict, reduce_options=args.reduce_options)["test"]
    if args.debug:
        test_dataset = test_dataset[:5]

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


    # CACHE_FNAME = "/home/sstorks/data/sstorks/MultimodalImaginationForPhysicalCausality/cache_files/trip_lm_cache_gpt4_verifiable_top-down.pkl"
    CACHE_FNAME = f'cache_files/trip_lm_cache_{args.lm_backbone}_{args.demo_choice}_{args.reasoning_depth}_{args.reasoning_direction}.pkl'
    if not os.path.exists('cache_files'):
        os.makedirs('cache_files')
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
    if args.reasoning_direction == "top-down":
        app_demos = generate_app_demos(train_dataset, reduce_options=args.reduce_options)
        aep_demos = generate_aep_demos(train_dataset, reduce_options=args.reduce_options)
    else:
        app_demos = generate_app_demos_fully_separate_familiarization(train_dataset, reduce_options=args.reduce_options)
        aep_demos = generate_aep_demos_fully_separate_familiarization(train_dataset, reduce_options=args.reduce_options)

    # print(app_demos)
    # print(aep_demos)

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
    print("selected story IDs:", selected_story_example_ids)
    story_demos = ""
    for sampled_train_data_i, selected_story_id in enumerate(selected_story_example_ids):
        # print([d['example_id'] for d in train_dataset])
        demo_example = [example for example in train_dataset if example['example_id'] == selected_story_id][0]
        story_pair = demo_example['stories']
        if args.reasoning_direction == 'top-down':
            if args.ask_implausible:
                story_demos += story_pair_demo_generator_topdown_ask_implausible(story_pair, 
                                                            conflict_reason_enabled=True if args.use_conflict_explanations else False,
                                                            reasoning_depth=args.reasoning_depth) + "\n"
            else:
                story_demos += story_pair_demo_generator_topdown(story_pair, 
                                                            conflict_reason_enabled=True if args.use_conflict_explanations else False,
                                                            reasoning_depth=args.reasoning_depth) + "\n"
        elif args.reasoning_direction == 'bottom-up':
            story_demos += story_pair_demo_generator_bottomup_compact(story_pair, 
                                                        conflict_reason_enabled=True if args.use_conflict_explanations else False,
                                                        reasoning_depth=args.reasoning_depth) + "\n"
       
    # print(story_demos)
    """
    # for debug: (# make sure change demonstration = story_pair_prompt_generator(story_A, story_B) ---> demonstration = "")
    prediction_obj = {}
    story_A, story_B = train_dataset[311]["stories"][0], train_dataset[311]["stories"][1]
    if story_A["plausible"] == True:
        implausible_story = story_B
        implausible_story_letter = "B"
    elif story_B["plausible"] == True:
        implausible_story = story_A
        implausible_story_letter = "A"
    add_trip_preds_bottomup_compact(prediction_obj, implausible_story, story_demos)
    print(prediction_obj)
    """
    
    attention_analysis_table = {}

    attention_analysis_table["layers"] = ["2nd Level Attention Ratio", "3rd Level Attention Ratio", "2nd Level Precision", "2nd Level Recall", "3rd Level Precision", "3rd Level Recall"]

    story_attn_thresholds = [0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12]
    sentences_attn_thresholds = [0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12]

    attention_analysis_metrics_over_thresholds = []

    if args.analyze_attention:

        for i in range(0, len(story_attn_thresholds)):

            story_attn_threshold = story_attn_thresholds[i]
            sentences_attn_threshold = sentences_attn_thresholds[i]

            predictions = []

            correct_story_total_attn = 0
            incorrect_story_total_attn = 0
            correct_sentences_total_attn = 0
            incorrect_sentences_total_attn = 0
            num_correct_story = 0
            num_correct_sentences = 0

            num_story_attn_exceed_threshold = 0
            num_sentences_attn_exceed_threshold = 0

            num_true_positive_second_level = 0
            num_true_negative_second_level = 0
            num_false_positive_second_level = 0
            num_false_negative_second_level = 0
            num_true_positive_third_level = 0
            num_true_negative_third_level = 0
            num_false_positive_third_level = 0
            num_false_negative_third_level = 0

            for i in tqdm(range(0, len(test_dataset))):

                example_id = test_dataset[i]['example_id']

                prediction_obj = {'example_id': example_id}
                stories_info = test_dataset[i]['stories']
                if stories_info[0]['plausible'] == True:
                    implausible_story = stories_info[1]
                else:
                    implausible_story = stories_info[0]
                story_prompt = story_pair_prompt_generator(stories_info[0], stories_info[1])

                # Prompt the LM
                if args.lm_backbone in ['gpt3', "llama7b", "llama13b", "llama30b", "llama65b"]:
                    prompt = ""
                    # To fully specify physical state prediction step, prepend several demos of it covering all physical attributes first
                    ##########################################################
                    # if you are using llama7b8bit, adding app demos and aep demos will be a problem caused repeated generation, try commenting the following:
                    if args.reasoning_depth == "verifiable":
                        prompt += app_demos + "\n" + aep_demos + "\n"
                    ##########################################################
                    prompt += story_demos + "\n" + story_prompt
                    # print("llm prompt:", prompt)
                    # print("llm prompt length:", len(tokenizer.encode(prompt)))
                    # Record prompt used with LM (splitting by newlines for readability)
                    prediction_obj['input_prompt'] = prompt.split('\n')

                    # if example_id not in lm_cache: # duplicate in prompt_gpt3_with_caching
                        # print("prompt:", prompt)
                    if args.lm_backbone == 'gpt3':
                        story_generated_text = prompt_gpt3_with_caching(prompt, args, lm_cache, example_id, max_tokens=128, 
                                                                        token_counts=TOKEN_COUNTS)
                    if args.lm_backbone in ["llama7b", "llama13b", "llama30b", "llama65b"]:
                        separations = separation_generator(app_demos + "\n" + aep_demos + "\n" + story_demos + "\n", story_prompt, tokenizer) if args.output_attn else None
                        story_generated_text = prompt_llama_with_caching(model, tokenizer, prompt, args, lm_cache, example_id, max_tokens=128, output_attn=args.output_attn, separations=separations)

                    # lm_cache[example_id] = story_generated_text
                    # pickle.dump(lm_cache, open(CACHE_FNAME, 'wb'))
                    # else:
                        # story_generated_text = lm_cache[example_id]

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

                    # To fully specify physical state prediction step, prepend several demos of it covering all physical attributes first
                    if args.reasoning_depth == "verifiable":
                        # Add some demos of APP and AEP specifically
                        app_demos_message = (
                            get_chat_message(
                                "system",
                                "Here are some examples of how you can understand physical *precondition* states in text:\n\n" + app_demos,
                                model_name=args.lm_backbone
                            )                
                        )
                        aep_demos_message = (
                            get_chat_message(
                                "system",
                                "Here are some examples of how you can understand physical *effect* states in text:\n\n" + aep_demos,
                                model_name=args.lm_backbone
                            )
                        )
                        messages += [app_demos_message, aep_demos_message]

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
                        story_generated_text = prompt_chat_gpt_with_caching(messages, args, lm_cache, example_id, max_tokens=128, token_counts=TOKEN_COUNTS)
                    elif args.lm_backbone in ['vicuna13b']:
                        story_generated_text = prompt_fastchat_with_caching(model, tokenizer, messages, args, lm_cache, example_id, max_tokens=128)

                # Gather up model predictions using templates and regular expressions
                prediction_obj['generated_text'] = story_generated_text.split('\n')
                if args.reasoning_direction == 'top-down':
                    if args.ask_implausible:
                        add_trip_preds_topdown_ask_implausible(prediction_obj, implausible_story, story_generated_text)
                    else:
                        add_trip_preds_topdown(prediction_obj, implausible_story, story_generated_text)
                else:
                    add_trip_preds_bottomup_compact(prediction_obj, implausible_story, story_generated_text)
                predictions.append(prediction_obj)
                # correct_story_total_attn
                # incorrect_story_total_attn
                # correct_sentences_total_attn
                # incorrect_sentences_total_attn
                # num_correct_story
                # num_correct_sentences
                correct_plausibility, correct_confl_pairs, correct_physical_states = trip_metrics_for_one_pred(test_dataset[i], prediction_obj)
                if correct_plausibility:
                    num_correct_story += 1
                    all_attn = np.array(lm_cache[example_id + "_attn"])
                    attn_mean = np.mean(all_attn[30:50], axis=0)
                    if stories_info[0]['plausible'] == True:
                        second_level_implausible_story_attn_mean = np.mean(attn_mean[1][1]) # first 1: second generated sentence; second 1: the second story -> implausible
                        second_level_plausible_story_attn_mean = np.mean(attn_mean[1][0])
                        third_level_effect_implausible_story_attn = attn_mean[3][1]
                        third_level_precondition_implausible_story_attn = attn_mean[5][1]
                    else:
                        second_level_implausible_story_attn_mean = np.mean(attn_mean[1][0]) # first 1: second generated sentence; second 0: the first story -> implausible
                        second_level_plausible_story_attn_mean = np.mean(attn_mean[1][1])
                        third_level_effect_implausible_story_attn = attn_mean[3][0]
                        third_level_precondition_implausible_story_attn = attn_mean[5][0]
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
                    confl_sentence_index_1, confl_sentence_index_2 = implausible_story["confl_pairs"][0]
                    third_level_conflict_sentences_attn_mean = (third_level_effect_implausible_story_attn[confl_sentence_index_1] + third_level_precondition_implausible_story_attn[confl_sentence_index_2]) / 2
                    third_level_irrelevant_sentences_attn_mean = (np.mean(np.delete(third_level_effect_implausible_story_attn, confl_sentence_index_1)) + np.mean(np.delete(third_level_precondition_implausible_story_attn, confl_sentence_index_2))) / 2
                    if correct_confl_pairs:
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
            
            precision_second_level = num_true_positive_second_level / (num_true_positive_second_level + num_false_positive_second_level)
            recall_second_level = num_true_positive_second_level / (num_true_positive_second_level + num_false_negative_second_level)
            f1_second_level = 2 * ((precision_second_level * recall_second_level) / (precision_second_level + recall_second_level))
            precision_third_level = num_true_positive_third_level / (num_true_positive_third_level + num_false_positive_third_level)
            recall_third_level = num_true_positive_third_level / (num_true_positive_third_level + num_false_negative_third_level)
            f1_third_level = 2 * ((precision_third_level * recall_third_level) / (precision_third_level + recall_third_level))

            # ["story ratio", "sentences ratio", "avg: # story exceeds threshold", "avg: # sentences exceeds threshold", "second level precision", "second level recall", "third level precision", "third level recall"]

            attention_analysis_metrics_over_thresholds.append([correct_story_total_attn / incorrect_story_total_attn, correct_sentences_total_attn / incorrect_sentences_total_attn, precision_second_level, recall_second_level, precision_third_level, recall_third_level])

            if not args.skip_prompting:
                metrics, predictions = trip_metrics(test_dataset, predictions)
                # pprint(metrics)
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
                output_dir, instance_name = get_output_dir_name('trip', args, instance_timestamp, 0, result_type="combined")
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
        
        attention_analysis_table_key = str(30) + " to " + str(50)
        attention_analysis_table[attention_analysis_table_key] = np.mean(np.vstack(attention_analysis_metrics_over_thresholds), axis=0)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame(attention_analysis_table)
        print(df)
    else:
        predictions = []
        for i in tqdm(range(0, len(test_dataset))):

            example_id = test_dataset[i]['example_id']

            prediction_obj = {'example_id': example_id}
            stories_info = test_dataset[i]['stories']
            if stories_info[0]['plausible'] == True:
                implausible_story = stories_info[1]
            else:
                implausible_story = stories_info[0]
            story_prompt = story_pair_prompt_generator(stories_info[0], stories_info[1])

            # Prompt the LM
            if args.lm_backbone in ['gpt3', "llama7b", "llama13b", "llama30b", "llama65b"]:
                prompt = ""
                # To fully specify physical state prediction step, prepend several demos of it covering all physical attributes first
                ##########################################################
                # if you are using llama7b8bit, adding app demos and aep demos will be a problem caused repeated generation, try commenting the following:
                if args.reasoning_depth == "verifiable":
                    prompt += app_demos + "\n" + aep_demos + "\n"
                ##########################################################
                prompt += story_demos + "\n" + story_prompt
                # print("llm prompt:", prompt)
                # print("llm prompt length:", len(tokenizer.encode(prompt)))
                # Record prompt used with LM (splitting by newlines for readability)
                prediction_obj['input_prompt'] = prompt.split('\n')

                # if example_id not in lm_cache: # duplicate in prompt_gpt3_with_caching
                    # print("prompt:", prompt)
                if args.lm_backbone == 'gpt3':
                    story_generated_text = prompt_gpt3_with_caching(prompt, args, lm_cache, example_id, max_tokens=128, 
                                                                    token_counts=TOKEN_COUNTS)
                if args.lm_backbone in ["llama7b", "llama13b", "llama30b", "llama65b"]:
                    separations = separation_generator(app_demos + "\n" + aep_demos + "\n" + story_demos + "\n", story_prompt, tokenizer) if args.output_attn else None
                    story_generated_text = prompt_llama_with_caching(model, tokenizer, prompt, args, lm_cache, example_id, max_tokens=128, output_attn=args.output_attn, separations=separations)

                # lm_cache[example_id] = story_generated_text
                # pickle.dump(lm_cache, open(CACHE_FNAME, 'wb'))
                # else:
                    # story_generated_text = lm_cache[example_id]

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

                # To fully specify physical state prediction step, prepend several demos of it covering all physical attributes first
                if args.reasoning_depth == "verifiable":
                    # Add some demos of APP and AEP specifically
                    app_demos_message = (
                        get_chat_message(
                            "system",
                            "Here are some examples of how you can understand physical *precondition* states in text:\n\n" + app_demos,
                            model_name=args.lm_backbone
                        )                
                    )
                    aep_demos_message = (
                        get_chat_message(
                            "system",
                            "Here are some examples of how you can understand physical *effect* states in text:\n\n" + aep_demos,
                            model_name=args.lm_backbone
                        )
                    )
                    messages += [app_demos_message, aep_demos_message]

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
                    story_generated_text = prompt_chat_gpt_with_caching(messages, args, lm_cache, example_id, max_tokens=128, token_counts=TOKEN_COUNTS)
                elif args.lm_backbone in ['vicuna13b']:
                    story_generated_text = prompt_fastchat_with_caching(model, tokenizer, messages, args, lm_cache, example_id, max_tokens=128)

            # Gather up model predictions using templates and regular expressions
            prediction_obj['generated_text'] = story_generated_text.split('\n')
            if args.reasoning_direction == 'top-down':
                if args.ask_implausible:
                    add_trip_preds_topdown_ask_implausible(prediction_obj, implausible_story, story_generated_text)
                else:
                    add_trip_preds_topdown(prediction_obj, implausible_story, story_generated_text)
            else:
                add_trip_preds_bottomup_compact(prediction_obj, implausible_story, story_generated_text)
            predictions.append(prediction_obj)

        if not args.skip_prompting:
            metrics, predictions = trip_metrics(test_dataset, predictions)
            # pprint(metrics)
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
            output_dir, instance_name = get_output_dir_name('trip', args, instance_timestamp, 0, result_type="combined")
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