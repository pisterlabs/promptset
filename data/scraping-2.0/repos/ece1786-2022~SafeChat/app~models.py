import torch
import numpy as np
import pandas as pd
import openai
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
from huggingface_hub import login

from app import webapp
from typing import List, Dict, Tuple


openai.api_key = webapp.config['OPENAI_KEY']
hugging_face_key = webapp.config['HUGGINGFACE_KEY']
login(hugging_face_key)


class Classification:

    def __init__(
        self, 
        model='Zero-shot',
        temperature=1.0, 
        max_length=16, 
        top_p=1.0, 
        frequency_penalty=0.0, 
        presence_penalty=0.0, 
        best_of=1,
        prompt=''
    ) -> None:

        if model == 'Zero-shot':
            selected_model = 'text-davinci-002'
        elif model == 'One/Few-shot':
            selected_model = 'text-davinci-002'
        else:
            selected_model = 'ada:ft-university-of-toronto-2022-12-03-23-52-58'

        self.configs = {
            "engine": selected_model,
            "temperature": temperature,
            "max_tokens": max_length,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "best_of": best_of
        }
        self.inputstart = "[TEXT]"
        self.outputstart = "[ANSWER]"
        self.prompt = prompt.strip()
        self.model_type = model

    def ask(self, input_text: str):
        if self.model_type in ['Zero-shot', 'One/Few-shot']:
            return self.ask_prompt(input_text)
        return self.ask_finetune(input_text)

    def ask_prompt(self, input_text: str):
        prompt_text = f"{self.prompt}\n\n{self.inputstart}: {input_text}\n{self.outputstart}: "
        response = openai.Completion.create(
            prompt=prompt_text,
            stop=[" {}:".format(self.outputstart)],
            logprobs=1,
            **self.configs
        )

        print(response)

        top_words = response["choices"][0]["logprobs"]["top_logprobs"][0]
        _, top_logprob = list(top_words.items())[0]
        text = response["choices"][0]["text"]

        return text, self._compute_prob_gpt3(top_logprob)

    def _compute_prob_gpt3(self, logprob: float) -> Tuple[float, float]:
        return 100 * np.e**logprob

    def ask_finetune(self, input_text: str, finetuned_version="ada:ft-university-of-toronto-2022-12-03-23-52-58"):
        def get_gpt3_result(text):
            if "non" in text.strip() and "suicide" in text.strip():
                return "non-suicide"
            elif "suicide" in text.strip():
                return "suicide"
            return "non-suicide"

        resp = openai.Completion.create(
            model=finetuned_version,
            prompt=input_text + "\n\n###\n\n",
            logprobs=1,
            max_tokens=self.configs.get("max_tokens"),
            temperature=self.configs.get("temperature")
        )
        print(f"resp: {resp}")
        top_words = resp["choices"][0]["logprobs"]["top_logprobs"][0]
        _, top_logprob = list(top_words.items())[0]
        text = resp["choices"][0]["text"]

        return get_gpt3_result(text), self._compute_prob_gpt3(top_logprob)


class GPT2Classification(Classification):

    def __init__(
        self, 
        model='Zero-shot',
        temperature=1.0, 
        max_length=16, 
        top_p=1.0, 
        num_beams=1,
        diversity_penalty=0.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        prompt=''
    ) -> None:

        if model == 'Zero-shot':
            selected_model = 'gpt2'
        elif model == 'One/Few-shot':
            selected_model = 'gpt2'
        else:
            selected_model = 'NoNameForMe/safechat-gpt2'

        # setup baseline model
        self.configs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_length": max_length,
            "diversity_penalty": diversity_penalty,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "num_beams": num_beams
        }

        # TODO: push a finetuned gpt2 image to Huggingface and then pull it here
        self.model_config = GPT2Config.from_pretrained(selected_model, num_labels=2, **self.configs) # Binary Classification
        self.model = GPT2ForSequenceClassification.from_pretrained(selected_model, config=self.model_config)

        self.tokenizer = GPT2Tokenizer.from_pretrained(selected_model)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.inputstart = "[TEXT]"
        self.outputstart = "[ANSWER]"
        self.prompt = prompt.strip()
        self.model_type = model
    
    def ask(self, input_text: str):
        if self.model_type in ['Zero-shot', 'One/Few-shot']:
            return self.ask_prompt(input_text)
        return self.ask_finetune(input_text)

    def ask_prompt(self, input_text: str):
        prompt_text = f"{self.prompt}\n\n{self.inputstart}: {input_text}\n{self.outputstart}: "
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        probs, indices = self._compute_prob_baseline(logits)
        prob = probs.item()
        cls = indices.item()
        if cls == 1:
            return "suicide", prob
        else:
            return "non-suicide", prob

    def _compute_prob_baseline(self, logits):
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        return torch.topk(probs, k=1)

    def ask_finetune(self, input_text: str):
        def get_gpt2_result(output):
            if output == 0:
                return 'non-suicide'
            return 'suicide'
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=None)
        outputs = self.model(**inputs)

        logits = outputs.logits
        probs, _ = self._compute_prob_baseline(logits)
        prob = probs.item()
        return get_gpt2_result(outputs["logits"].argmax(axis=-1).item()), prob * 100
