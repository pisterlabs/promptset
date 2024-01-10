import argparse
import re

import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Union, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from api import BaseAPI
from utils import _extract_api_request_content
from prompt import PromptTemplate
        
from OpenAIModels import OpenAIGPT3, OpenAIChatGPT
import time

CACHE_DIR="/scratch/ylu130/huggingface/models"

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], PromptLength=0):
        super().__init__()
        self.stops = stops
        self.non_stop_count = 0
        self.PromptLength = PromptLength

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            stop_count_list = [(stop == input_ids[i][self.PromptLength:]).sum().item() for i in range(input_ids.shape[0])]
        
        # at least one token is different from the stop token
        non_stop_count_list = [(stop != input_ids[i][self.PromptLength:]).sum().item() for i in range(input_ids.shape[0])]

        if all(stop_count > 1 for stop_count in stop_count_list) and all(non_stop_count > 1 for non_stop_count in non_stop_count_list):
            return True
        return False

class OpenAI_Zero_Executor:
    def __init__(self, args: argparse.Namespace, prompt: PromptTemplate = None) -> None:
        self.verbose = args.verbose
        self.prompt = prompt
        self.args = args

    def execute(self, input: str, prompt: Optional[PromptTemplate] = None) -> Optional[str]:
        """
        Put input question and prompt into language model and return the generated output
        Args:
            input (str): input question
        Returns:
            Optional[str]: output from language model
        """
        if self.verbose:
            if prompt is None:
                print(f"***********************OpenAI Zero-Shot Start Executing***********************")
            else:
                print(f"***********************OpenAI Few-Shot Start Executing***********************")
        if self.args.openai_model == "chatgpt":
            model = OpenAIChatGPT(prompt if prompt is not None else self.prompt)
        else: 
            model = OpenAIGPT3(prompt if prompt is not None else self.prompt)
        
        model.append_message_from_user(input)

        trying = 0
        while trying < 3:
            try:
                output = model.send_feedback()
                break
            except Exception as e:
                print("OpenAI API call failed, reason: ", e)
                output = None
                time.sleep(5+trying**2)
                trying += 1
        if self.verbose:
            print(f"OpenAI API Input: {input}")
            print(f"OpenAI API Response: {output}")
            print(f"***********************End Executing***********************")
        return output

class OpenAIAPIExecutor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.verbose = args.verbose
        self.args = args

    def call(self, input: str, apis: List[BaseAPI]) -> List[Optional[str]]:
        """
        Put input question and prompt into language model and return the generated output
        Args:
            input (str): input question
        Returns:
            List[str]: output from language model
        """
        outputs = []
        for api in apis:
            if api.prompt_template != None:
                if self.args.openai_model == "chatgpt":
                    model = OpenAIChatGPT(api.prompt_template)
                else:
                    model = OpenAIGPT3(api.prompt_template)
                model.append_message_from_user(input)
                trying = 0
                while trying < 3:
                    try:
                        output = model.send_feedback()
                        break
                    except Exception as e:
                        print("OpenAI API call failed, reason: ", e)
                        output = None
                        time.sleep(5+trying**2)
                        trying += 1
                outputs.append(output)
            else:
                output = None
                outputs.append(output)
            if self.verbose:
                print(f"{api.name} API Call: {output}")
        return outputs
    
    def obtain_api_response(self, input: str, outputs: List[str], apis: List[BaseAPI]) -> List[Optional[str]]:
        """
        Obtain the response from generated outputs
        """
        api_responses = []

        for api, output in zip(apis, outputs):
            if api.prompt_template is None: # APIs do not need API calls from LM
                request_args = None
                if api.name != "MultilingualQA":
                    api_response = api(input)
                else:
                    try:
                        src =  api.apis['MT'].translator.detect(re.search('question: (.*)context:', input).group(1)).lang if re.search('question: (.*)context:', input) is not None else 'en'
                        api_response = api(input, mtdest = 'en', mtsrc = src)
                    except: 
                        api_response = None
            else:                           # APIs need generated API calls
                request_args = _extract_api_request_content(output, api.name)
                if request_args is None:
                    api_response = None
                else:
                    try:
                        api_response = api(*request_args)
                    except: # MT missing positional argument 'mtdest'
                        api_response = None

            if self.verbose:
                print(f"{api.name} request content: {request_args}")
                print(f"{api.name} response: {api_response}")
            api_responses.append(api_response)
        return api_responses

    def execute(self, input: str, apis: List[BaseAPI]) -> Tuple[BaseAPI, Optional[str]]:
        """
        Filter the APIs by the cross-entropy between the generated outputs and the input question
        Args:
            input (str): input question
        Returns:
            Tuple[BaseAPI, Optional[str]]: the selected API and its response
        """

        if self.verbose:
            print(f"***********************OpenAI Start Executing***********************")
        generated_outputs = self.call(input, apis)

        api_responses = self.obtain_api_response(input, generated_outputs, apis)

        # remove the APIs that have unmeaningful responses
        for api, api_response in zip(apis, api_responses):
            if api_response is None or api_response.lower().strip() in ["", "unknown", "none", "no answer"]:
                apis.remove(api)
                api_responses.remove(api_response)
        
        if len(apis) == 0:
            # no suitable API found, return the answer from the LM
            if self.args.openai_model == "chatgpt":
                model = OpenAIChatGPT()
            else:
                model = OpenAIGPT3()
            model.append_message_from_user(input)
            trying = 0
            while trying < 3:
                try:
                    output_without_prompt = model.send_feedback()
                    break
                except Exception as e:
                    print("OpenAI API call failed, reason: ", e)
                    output_without_prompt = None
                    trying += 1

            if self.verbose:
                print(f"answer: {output_without_prompt} selected_api: {None}")
                print(f"***********************End Executing***********************")

            return None, output_without_prompt
        else:
            # only one suitable API found, return the answer from the API
            if self.verbose:
                print(f"answer: {api_responses[0]} selected_api: {apis[0].name}")
                print(f"***********************End Executing***********************")

            return apis[0].name, api_responses[0]


class LMAPIExecutor:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device(args.fdevice if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModelForCausalLM.from_pretrained(args.llm, cache_dir=CACHE_DIR).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm, padding_side="left", cache_dir=CACHE_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.model.eval()

        self.max_tokens = args.max_tokens
        self.verbose = args.verbose

    def _compute_weight(self, t: int) -> Union[int, float]:
        """Compute the weight in the loss function."""
        return max(0, 1-0.25*t)

    def _normalize_weights(self, augmented_text_ids):
        """Normalize the weight of each position in a sequence."""
        for api in augmented_text_ids.values():
            total_weight = sum([seq_position["unnormalized_weight"] for seq_position in api["seq_positions"].values()])
            for seq_position in api["seq_positions"].values():
                seq_position["normalized_weight"] = seq_position["unnormalized_weight"] / total_weight
        
        return augmented_text_ids
    
    def _extract_conditioning_ids_and_target_ids(self, augmented_text_ids):
        conditioning_text_ids = torch.tensor([])
        target_ids = torch.tensor([])
        
        for _, api_dict in augmented_text_ids.items():
            for _, seq_position_dict in api_dict["seq_positions"].items():
                target_ids = torch.concat([target_ids, seq_position_dict["target_ids"]], dim=0)
                conditioning_text_ids = torch.cat([
                    conditioning_text_ids,
                    F.pad(seq_position_dict["prompt_ids"].long(), pad=(self.max_tokens-seq_position_dict["prompt_ids"].shape[-1], 0), value=self.tokenizer.pad_token_id).unsqueeze(0)
                ], dim=0)
    
        return conditioning_text_ids.long(), target_ids.long()

    def _extract_target_logprob_from_logits(self, logits, target_ids):
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs[range(target_ids.shape[-1]), target_ids]
        return target_log_probs
    
    def _calculate_cross_entropy(self, augmented_text_ids):
        cross_entropy = {}
        for api_name in augmented_text_ids:        
            cross_entropy[api_name] = 0
            seq_positions = augmented_text_ids[api_name]["seq_positions"]
            for i in seq_positions:
                cross_entropy[api_name]  += -(seq_positions[i]["losses"] * seq_positions[i]["normalized_weight"])
        
        return cross_entropy
    
    def call(self, input: str, apis: List[BaseAPI]) -> List[Optional[str]]:
        outputs = []
        eos_token_id = self.tokenizer("\n")["input_ids"][0] # eos_token_id list for EleutherAI/gpt-neo-1.3B

        prompts_batch = [api.prompt_template.format(input=input) for api in apis if api.prompt_template is not None]
        if len(prompts_batch) != 0:
            inputs_batch = self.tokenizer(prompts_batch, padding=True, return_tensors="pt")
            PROMPT_LENGTH = len(inputs_batch['input_ids'][0])
            # do sampling because LM2 is powerful
            outputs_ids = self.model.generate(inputs_batch['input_ids'].to(self.device),
                                              attention_mask=inputs_batch['attention_mask'].to(self.device),
                                              do_sample=True,
                                              top_k=30,
                                              top_p=0.9,
                                              eos_token_id = eos_token_id,
                                              pad_token_id = self.tokenizer.eos_token_id, 
                                              max_new_tokens = self.max_tokens)
            outputs_ids = outputs_ids[:, PROMPT_LENGTH:]
            outputs = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)

        if len(prompts_batch) != len(apis):
            outputs.extend([None] * (len(apis) - len(outputs)))
        
        if self.verbose:
            for i in range(len(apis)):
                print(f"{apis[i].name} API Call: {outputs[i]}")

        return outputs
    
    def obtain_api_response(self, input: str, outputs: List[str], apis: List[BaseAPI]) -> List[Optional[str]]:
        """
        Obtain the response from generated outputs
        """
        api_responses = []

        for api, output in zip(apis, outputs):
            if api.prompt_template is None: # APIs do not need API calls from LM
                request_args = None
                if api.name != "MultilingualQA":
                    api_response = api(input)
                else:
                    try:
                        src =  api.apis['MT'].translator.detect(re.search('question: (.*)context:', input).group(1)).lang if re.search('question: (.*)context:', input) is not None else 'en'
                        api_response = api(input, mtdest = 'en', mtsrc = src)
                    except: 
                        api_response = None
            else:                           # APIs need generated API calls
                request_args = _extract_api_request_content(output, api.name)
                if request_args is None:
                    api_response = None
                else:
                    try:
                        api_response = api(*request_args)
                    except: # MT missing positional argument 'mtdest'
                        api_response = None

            if self.verbose:
                print(f"{api.name} request content: {request_args}")
                print(f"{api.name} response: {api_response}")
            api_responses.append(api_response)
        
        return api_responses


    def execute(self, input: str, apis: List[BaseAPI]) -> Tuple[BaseAPI, Optional[str]]:
        """
        Filter the APIs by the cross-entropy between the generated outputs and the input question
        Args:
            input (str): input question
        Returns:
            Tuple[BaseAPI, Optional[str]]: the selected API and its response
        """

        apis.sort(key=lambda x: x.prompt_template is None)

        if self.verbose:
            print(f"***********************GPTJ Start Executing***********************")
        generated_outputs = self.call(input, apis)

        api_responses = self.obtain_api_response(input, generated_outputs, apis)

        # remove the APIs that have unmeaningful responses
        for api, api_response in zip(apis, api_responses):
            if api_response is None or api_response.lower().strip() in ["", "unknown", "none", "no answer"]:
                apis.remove(api)
                api_responses.remove(api_response)
        
        if len(apis) == 0:
            # no suitable API found, return the answer from the LM
            input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
            eos_token_id = self.tokenizer(".\n")["input_ids"][0] 

            with torch.no_grad():
                output_ids_without_prompt = self.model.generate(input_ids=input_ids,
                                                                   do_sample = True,
                                                                   top_p = 0.9,
                                                                   top_k = 20,
                                                                   eos_token_id=eos_token_id,
                                                                   pad_token_id = self.tokenizer.eos_token_id,
                                                                   max_new_tokens =self.max_tokens)

            output_ids_without_prompt = output_ids_without_prompt[:, input_ids.shape[1]:]
            output_without_prompt = self.tokenizer.decode(output_ids_without_prompt[0], skip_special_tokens=True).strip()

            if self.verbose:
                print(f"answer: {output_without_prompt} selected_api: {None}")
                print(f"***********************End Executing***********************")

            return None, output_without_prompt
        elif len(apis) == 1:
            # only one suitable API found, return the answer from the API
            if self.verbose:
                print(f"answer: {api_responses[0]} selected_api: {apis[0].name}")
                print(f"***********************End Executing***********************")

            return apis[0].name, api_responses[0]
        else:
            input_ids = self.tokenizer(input + "The answer is: ", return_tensors="pt")["input_ids"][0]
            augmented_text_ids = {}

            for api, api_response in zip(apis, api_responses):
                # test if the API response is None
                if api_response is None:   # skip when the API call cannot be executed (return an empty string)
                    continue
                else:
                    api_response_ids = self.tokenizer(api_response, return_tensors="pt")["input_ids"][0]
                    # concatenate the input question and the API response
                    question_response_ids = torch.cat((input_ids, api_response_ids), dim=0)
                    augmented_text_ids[api.name] = {"seq_positions": {}}
                    # j start from the API response
                    j = len(input_ids)
                    while j < len(question_response_ids) and self._compute_weight(t=j-len(input_ids)) > 0:
                        conditioning_text_ids = question_response_ids[:j]
                        next_token_ids = question_response_ids[j]

                        augmented_text_ids[api.name]["seq_positions"][j] = {
                            "prompt_ids": conditioning_text_ids,
                            "unnormalized_weight": self._compute_weight(t=j-len(input_ids)),
                            "losses": [],
                            "target_ids": torch.tensor([next_token_ids])
                        }
                        j += 1

            # augmented_text_ids: {api: {seq_position: {prompt_ids, unnormalized_weight, losses, target_ids}}}
            augmented_text_ids = self._normalize_weights(augmented_text_ids)
            # conditioning_text_ids.shape: [# total tokens in responses, max_tokens]
            # target_ids.shape: [# total tokens in responses]
            conditioning_text_ids, target_ids = self._extract_conditioning_ids_and_target_ids(augmented_text_ids)

            if self.verbose:
                print(f"Total number of tokens in API responses (batch): {conditioning_text_ids.shape[0]}")

            # if batch was too large, split into smaller batches
            with torch.no_grad():
                output = self.model(input_ids=conditioning_text_ids.to(self.device))
            logits = output.logits[:, -1, :].to("cpu")

            # log_probs: [3 * #j]
            log_probs = self._extract_target_logprob_from_logits(logits, target_ids)
            for _, api_dict in augmented_text_ids.items():
                for _, seq_position_dict in api_dict["seq_positions"].items():
                    seq_position_dict["losses"] = log_probs[:1].squeeze(0)
                    log_probs = log_probs[1:]

            cross_entropy = self._calculate_cross_entropy(augmented_text_ids)

            #  find the api in cross_entropy with the lowest cross entropy
            selected_api = min(cross_entropy, key=cross_entropy.get)
            for index, api in enumerate(apis):
                if api.name == selected_api:
                    response = api_responses[index]
                    break

            if self.verbose:
                print(cross_entropy)
                print(f"answer: {response} selected_api: {selected_api}")
                print(f"***********************End Executing***********************")

            return selected_api, response 

    def zero_shot_batch(self, input: List[str], batch_size: int) -> List[str]:

        inputs_batch_list = [input[i:i + batch_size] for i in range(0, len(input), batch_size)]
        # eos_token_id = self.tokenizer("\n")["input_ids"]
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in ["\n"]]
        outputs_list = []
       
        for inputs_batch in inputs_batch_list:

            inputs_batch = self.tokenizer(inputs_batch, padding=True, return_tensors="pt")
            PROMPT_LENGTH = len(inputs_batch['input_ids'][0])
            
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, PromptLength=PROMPT_LENGTH)])
            outputs_ids = self.model.generate(inputs_batch['input_ids'].to(self.device),
                                             attention_mask=inputs_batch['attention_mask'].to(self.device),
                                             do_sample=True,
                                             top_k=30,
                                             top_p=0.9,
                                             stopping_criteria=stopping_criteria,
                                             pad_token_id = self.tokenizer.eos_token_id,
                                             max_new_tokens = self.max_tokens)
            outputs_ids = outputs_ids[:, PROMPT_LENGTH:]
            outputs = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
            
            outputs = [output.strip().split("\n")[0] for output in outputs]

            outputs_list.extend(outputs)

        return outputs_list
                                         