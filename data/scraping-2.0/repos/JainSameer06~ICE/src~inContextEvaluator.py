import os
import json
import random
import openai
import argparse
import re
import time
import math
from scipy.stats import spearmanr, pearsonr, kendalltau
from constants import *
from transformers import GPT2TokenizerFast
import sys
import signal

openai.api_key = os.environ["OPENAI_API_KEY"]
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)

class InContextEvaluator:
    def __init__(self, example_file, test_data_file, results_dir, task_prompt_file, mode, seed, fold_id, aspect, task, exclude_reference, exclude_description, sample=True, sample_size=None, max_tokens=30, num_candidates=3):
        self.task = task
        self.aspect = aspect

        # Prompt settings
        # By default, do not include text or reference; use this setting for fluency and coherence
        self.include_text = False
        self.include_reference = False  
        self.mode = mode

        if self.aspect == "consistency":    # use text for consistency
            self.include_text = True
        
        if self.aspect == "relevance":      
            if not exclude_reference:
                self.include_reference = True       # use reference for relevance by default
            else:
                self.include_text = True            # if exclude_reference is specified, include text instead
                
        # Test data sampling settings
        self.sample = sample
        self.fold_id = fold_id
        self.sample_size = sample_size
        self.seed = seed
        random.seed(self.seed)

        # Data and examples
        self.in_context_examples = self.getIncontextExamples(example_file)
        self.test_data = self.getTestData(test_data_file)

        # Prompt segments
        self.exclude_description = exclude_description
        if exclude_description:
            self.task_prompt = ""
        else:
            self.task_prompt = self.getTaskPrompt(task_prompt_file)
        self.question_prompt = "Is the following summary {}?".format(self.getAdjective(aspect))
        self.filler_prompt = "Given the following text"

        # GPT-3 settings
        self.num_candidates = num_candidates
        self.max_tokens = max_tokens

        # Results directory
        self.results_dir = results_dir

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def getIncontextExamples(self, example_file):
        with open(example_file) as f_examples:
            examples = json.load(f_examples)

        if self.mode == "single":
            random.shuffle(examples)
        if self.mode == "multi":
            for example in examples:
                random.shuffle(example)
            if self.include_text:
                examples = examples[:3]     # multi-example setting might cross the token limit
                                            # thus reducing the number of (outer) examples to 3
        return examples

    def getAdjective(self, aspect):
        adjective_map = {"fluency": "fluent", "coherence": "coherent", "relevance": "relevant",
                         "consistency": "consistent"}
        return adjective_map[aspect]

    def getTestData(self, test_data_file):
        '''The method assumes that test data and in-context examples have been separated at the 
        data preparation stage to ensure no overlap. It does not check for an overlap'''
        with open(test_data_file) as f_test:
            return json.load(f_test)

    def getTaskPrompt(self, task_prompt_file):
        '''Returns task description for the given aspect'''
        with open(task_prompt_file) as f_prompt:
            return json.load(f_prompt)[self.task][self.aspect]

    def buildDialogPrompt(self, test_example):
        '''Returns the complete prompt string to be passed to the model for a single test example 
        when the settings are set to expect numerical response'''
        text_prompts = []
        built_examples = []

        # Construct in-context example prompts
        for example in self.in_context_examples:
            aspect_score_expert = float(example["expert_{}_mean".format(self.aspect)])
            if self.include_text:
                example_str = '\nContext: {}\nResponse: {}\n{}: {}\n'.format(example["context"], example["response"], self.aspect.title(), str(aspect_score_expert))
            else:
                example_str = '\Response: {}\n{}: {}\n'.format(example["context"], self.aspect.title(), str(aspect_score_expert))
            built_examples.append(example_str)
        example_context = "".join(built_examples)

        # Construct test example prompt
        if self.include_text:
            built_test_example = '\Content: {}\Response: {}\n{}:'.format(test_example["context"], test_example["response"], self.aspect.title())
        else:
            built_test_example = '\Response: {}\n{}:'.format(test_example["response"], self.aspect.title())

        # Combine all prompt segments
        complete_prompt = "{}\n{}{}".format(self.task_prompt, example_context, built_test_example)
        print(complete_prompt)
        #sys.exit()
        return complete_prompt

    def buildSummarizationSinglePrompt(self, test_example):
        '''Returns the complete prompt string to be passed to the model for a single test example 
        when the settings are set to single example per text'''
        text_prompts = []
        built_examples = []

        # Construct in-context example prompts
        for example in self.in_context_examples:
            aspect_score_expert = float(example["expert_{}_mean".format(self.aspect)])
            if self.include_text:
                example_str = '\nText: {}\nSummary: {}\n{}: {}\n'.format(example["text"], example["decoded"], self.aspect.title(), str(aspect_score_expert))
            elif self.include_reference:
                example_str = '\nReference: {}\nSummary: {}\n{}: {}\n'.format(example["references"][0], example["decoded"], self.aspect.title(), str(aspect_score_expert))                 
            else:
                example_str = '\nSummary: {}\n{}: {}\n'.format(example["decoded"], self.aspect.title(), str(aspect_score_expert))
            built_examples.append(example_str)
        example_context = "".join(built_examples)

        # Construct test example prompt
        if self.include_text:
            built_test_example = '\nText: {}\nSummary: {}\n{}:'.format(test_example["text"], test_example["decoded"], self.aspect.title())
        elif self.include_reference:
            built_test_example = '\nReference: {}\nSummary: {}\n{}:'.format(test_example["references"][0], test_example["decoded"], self.aspect.title())
        else:
            built_test_example = '\nSummary: {}\n{}:'.format(test_example["decoded"], self.aspect.title())            

        # Combine all prompt segments
        complete_prompt = "{}\n{}{}".format(self.task_prompt, example_context, built_test_example)
        print(complete_prompt)
        # sys.exit()
        return complete_prompt

    def buildSummarizationMultiPrompt(self, test_example):
        '''Returns the complete prompt string to be passed to the model for a single test example
        when the settings are set to multiple examples per text'''
        text_prompts = []
        built_examples = []

        # Construct in-context example prompts
        for example in self.in_context_examples:
            # TODO: Add include_reference
            if self.include_text:
                example_str = '\nText: {}\n{}'.format(example[0]["text"], "".join(['Summary: {}\n{}: {}\n'.format(e["decoded"],
                                                                         self.aspect.title(),
                                                                         str(float(e["expert_{}_mean".format(self.aspect)]))) for e in example]))

            # if self.include_reference:
            #     example_str = '\nReference: {}\n{}'.format(example[0]["references"][0], "".join(['Summary: {}\n{}: {}\n'.format(e["decoded"],
            #                                                              self.aspect.title(),
            #                                                              str(float(e["expert_{}_mean".format(self.aspect)]))) for e in example]))  

            elif self.include_reference:
                example_str = '{}'.format("".join(['\nReference: {}\nSummary: {}\n{}: {}\n'.format(example[0]["references"][0], e["decoded"],
                                                                         self.aspect.title(),
                                                                         str(float(e["expert_{}_mean".format(self.aspect)]))) for e in example]))                                                          
            else:
                example_str = "".join(['\nSummary: {}\n{}: {}\n'.format(e["decoded"],
                                                                         self.aspect.title(),
                                                                         str(float(e["expert_{}_mean".format(self.aspect)]))) for e in example])
            built_examples.append(example_str)
        example_context = "".join(built_examples)

        # Construct test example prompt
        if self.include_text:
            built_test_example = '\nText: {}\nSummary: {}\n{}:'.format(test_example["text"], test_example["decoded"], self.aspect.title())
        elif self.include_reference:
            built_test_example = '\nReference: {}\nSummary: {}\n{}:'.format(test_example["references"][0], test_example["decoded"], self.aspect.title())
        else:
            built_test_example = '\nSummary: {}\n{}:'.format(test_example["decoded"], self.aspect.title())

        # Combine all prompt segments
        complete_prompt = "{}\n{}{}".format(self.task_prompt, example_context, built_test_example)
        # print(complete_prompt)
        # sys.exit()
        return complete_prompt
    
    def buildSummarizationPrompt(self, test_example):
        if self.mode == "single":
            return self.buildSummarizationSinglePrompt(test_example)
        if self.mode == "multi":
            return self.buildSummarizationMultiPrompt(test_example)

    def getTokenIds(self, tokens:list):
        token_ids = []
        for token in tokens:
            token_ids.extend(tokenizer.encode(token))
        return token_ids

    def getModelResponse(self, prompt):
        model = "text-davinci-003"
        frequency_penalty = 0
        presence_penalty = 0
        logprobs = 5

        #time.sleep(1)
        return openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0,
            max_tokens=self.max_tokens,
            n=self.num_candidates,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs
        )

    def runTest(self):
        responses = []
        for test_example in self.test_data:
            if self.task == "dialog":
                prompt = self.buildDialogPrompt(test_example)
            elif self.task == "summarization":
                prompt = self.buildSummarizationPrompt(test_example)
            else: 
                raise ValueError("Task type should be either dialog or summarization. {} was specified instead".format(self.experiment))
            
            num_attempts = 0
            while True:
                num_attempts += 1
                if num_attempts >= 20:
                    print("Too many attempts. Exiting.")
                    sys.exit()
                signal.alarm(10)    

                try:
                    response = self.getModelResponse(prompt)
                except TimeoutException as te:
                    print("Caught timeout on iteration {}. Retrying \n".format(i))
                    continue # continue the for loop if function A takes more than 5 second
                except Exception as e:
                    print("General exception caught. Retrying \n")
                    continue
                else:
                    signal.alarm(0)
                
                responses.append(response)
                break
        
        completions_with_scores = self.populateAspectScoresContinuous(responses)
        correlations = self.metaEvaluate(completions_with_scores)
        self.saveResponses(responses, completions_with_scores, correlations)

    def examplesWithPredictions(self, completions):
        assert len(completions) == len(self.test_data)
        for test_example, completion in zip(self.test_data, completions):
            test_example["score_{}".format(self.aspect)] = completion["positive_prob"]
        return self.test_data

    def saveResponses(self, responses, completions, correlations):
        """
        - When the setting is set to analyzing the impact of different ICEs, the seed will almost
          certainly be set to 1 (and is insignificant), and fold_id will correspond to ICE fold IDs
        - When the setting is set to analyzing the impact of ICE orderings, the fold_id will be 
          held constant (to the value of the ICE fold used), and the seed will be significant
        - When the setting is complete cross-validation, the seed will almost certainly be set
          to 1 (and is insignificant), and fold_id will correspond to test example fold IDs
        """

        if self.exclude_description:
            description_flag = "nd"
        else:
            description_flag = "d"

        complete_response_filename = "response_os{}_f{}_{}.json".format(self.seed, self.fold_id, description_flag)
        complete_response_file = os.path.join(self.results_dir, complete_response_filename)

        completions_filename = "completions_os{}_f{}_{}.json".format(self.seed, self.fold_id, description_flag)
        completions_file = os.path.join(self.results_dir, completions_filename)

        correlations_filename = "correlations_os{}_f{}_{}.json".format(self.seed, self.fold_id, description_flag)
        correlations_file = os.path.join(self.results_dir, correlations_filename)
        
        with open(complete_response_file, mode="w") as f_response:
            json.dump(responses, f_response, indent=1)

        with open(completions_file, mode="w") as f_completions:
            json.dump(self.examplesWithPredictions(completions), f_completions, indent=1)
        
        with open(correlations_file, mode="w") as f_correlations:
            json.dump(correlations, f_correlations, indent=1)

    def metaEvaluate(self, completions):
        expert_score_key = "expert_{}_mean".format(self.aspect)
        # turker_score_key = "turker_{}_mean".format(self.aspect)

        annotations = {}
        annotations["expert"] = [example[expert_score_key] for example in self.test_data]
        # annotations["turker"] = [example[turker_score_key] for example in self.test_data]
        predicted_scores = [completion["positive_prob"] for completion in completions]

        print("expert", annotations["expert"])
        # print("turker", annotations["turker"])
        print("predictions", predicted_scores)

        expert_correlations = {}
        expert_correlations["pearson"] = pearsonr(predicted_scores, annotations["expert"])[0]
        expert_correlations["spearman"] = spearmanr(predicted_scores, annotations["expert"])[0]
        expert_correlations["kendall"] = kendalltau(predicted_scores, annotations["expert"])[0]

        # turker_correlations = {}
        # turker_correlations["pearson"] = pearsonr(predicted_scores, annotations["turker"])[0]
        # turker_correlations["spearman"] = spearmanr(predicted_scores, annotations["turker"])[0]
        # turker_correlations["kendall"] = kendalltau(predicted_scores, annotations["turker"])[0]

        # correlations = {"expert": expert_correlations, "turker": turker_correlations}
        correlations = {"expert": expert_correlations}
        return correlations

    def populateAspectScoresContinuous(self, responses):
        completions = []
        for response in responses:
            response_id = response["id"]
            response_text = response["choices"][0]["text"]  # considering only the top choice
            response_text = re.findall('\d*\.?\d+', response_text)[0]
            positive_prob = float(response_text.strip())
            negative_prob = 1 - positive_prob
            completions.append({
                "response_id": response_id, "positive_prob": positive_prob, "negative_prob": negative_prob,
                "response": response_text
            })
        return completions

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--example_file', required=True)
    parser.add_argument('--test_data_file', required=True)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--task_prompt_file', required=True)
    parser.add_argument('--aspect', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--max_tokens', required=True, type=int)
    parser.add_argument('--num_candidates', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--fold_id', required=True, type=int)
    parser.add_argument('--sample', default=False, action='store_true',
                        help="True if sampling the test data; specify sample_size in this case")
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--task', required=True)
    parser.add_argument('--exclude_reference', default=False, action='store_true',
                        help="Flag for performing relevance evaluation in summarization without reference")
    parser.add_argument('--exclude_description', default=False, action='store_true',
                        help="Flag for excluding task description at the top of the prompt")


    args = parser.parse_args()

    ice = InContextEvaluator(
        example_file=args.example_file,
        test_data_file=args.test_data_file,
        results_dir=args.results_dir,
        task_prompt_file=args.task_prompt_file,
        aspect=args.aspect,
        mode=args.mode,
        seed=args.seed,
        fold_id=args.fold_id,
        sample=args.sample,
        sample_size=args.sample_size,
        max_tokens=args.max_tokens,
        num_candidates=args.num_candidates,
        task=args.task,
        exclude_reference=args.exclude_reference,
        exclude_description=args.exclude_description
    )
    ice.runTest()
