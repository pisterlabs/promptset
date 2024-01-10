import sys
sys.path.append('..')
import hydra
from typing import Optional, Tuple, List
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          GPT2LMHeadModel, AutoModelForCausalLM,
                          DebertaTokenizer, DebertaForMaskedLM,
                          BertTokenizer, BertForMaskedLM)

SUPPORTED_LEFT_TO_RIGHT_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                               'gpt2-large', 'gpt2-xl']
SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large', 'deberta-large', 'bert-large-cased']
import openai
import math
import concurrent.futures
import time


class PromptedClassificationEvaluator:
    def __init__(
        self,
        task_lm: str,
        is_mask_lm: Optional[bool],
        num_classes: int,
        verbalizers: List[str],
        template: Optional[str],
        template_trigger: Optional[str],
        prompt: str,
        trigger: str,
        target: int
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.task_lm = task_lm
        print("Task LM:", self.task_lm)
        if is_mask_lm is None:
            # If False, then treat as left-to-right LM
            self.is_mask_lm = True if 'bert' in self.task_lm else False
        else:
            self.is_mask_lm = is_mask_lm
        if self.task_lm == 'deberta-large':
            self._tokenizer = DebertaTokenizer.from_pretrained('lsanochkin/deberta-large-feedback')
            self._generator = (DebertaForMaskedLM
                               .from_pretrained('lsanochkin/deberta-large-feedback')
                               .to(self.device))
        elif self.task_lm == 'bert-large-cased':
            self._tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
            self._generator = (BertForMaskedLM
                               .from_pretrained('bert-large-cased')
                               .to(self.device))
        elif self.task_lm == 'gpt-j':
            self._tokenizer = AutoTokenizer.from_pretrained(
                'EleutherAI/gpt-j-6B', pad_token='<|endoftext|>',
                revision="float16", torch_dtype=torch.float16
            )
            self._generator = (AutoModelForCausalLM.from_pretrained(
                'EleutherAI/gpt-j-6B', revision="float16", torch_dtype=torch.float16,
            ).to(self.device))
        elif self.task_lm in ["gpt3.5", "gpt3"]:
            openai.api_key = "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"
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2", pad_token='<|endoftext|>')
            self._generator = (GPT2LMHeadModel.from_pretrained("gpt2").to(self.device))
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id
        elif self.task_lm == "llama-2-7b":
            self._tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            self._tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self._generator = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(self.device)
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id
            self._generator.resize_token_embeddings(len(self._tokenizer))
        elif self.task_lm == "llama-2-13b":
            self._tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
            self._tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self._generator = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf").to(self.device)
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id
            self._generator.resize_token_embeddings(len(self._tokenizer))
        elif self.is_mask_lm:
            assert self.task_lm in SUPPORTED_MASK_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm,
                                truncation_side="left")
            self._generator = (AutoModelForMaskedLM
                               .from_pretrained(self.task_lm)
                               .to(self.device))
        else:
            assert self.task_lm in SUPPORTED_LEFT_TO_RIGHT_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.task_lm, pad_token='<|endoftext|>')
            self._generator = (GPT2LMHeadModel
                               .from_pretrained(self.task_lm)
                               .to(self.device))
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id
        self.num_classes = num_classes
        self.verbalizers = verbalizers
        if self.task_lm not in ["gpt3.5", "gpt3"]:
            self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v) for v in self.verbalizers]
        if template is None or template_trigger is None:
            self.template, self.template_trigger = self.load_default_template()  # prompt templates
        else:
            self.template, self.template_trigger = template, template_trigger

        self.prompt = prompt
        self.trigger = trigger
        self.target = target

    # Adapted from
    # https://huggingface.co/docs/transformers/v4.21.1/en/task_summary#masked-language-modeling
    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(
            input_ids == self._tokenizer.mask_token_id)[1]
        return mask_token_index

    def load_default_template(self) -> Tuple[str, Optional[str]]:
        if self.task_lm in ['deberta-large', 'bert-large-cased']:
            template = "{sentence_1} {prompt} [MASK] ."
            template_trigger = "{sentence_1}{trigger} {prompt} [MASK] ."
        elif self.is_mask_lm:
            template = "{sentence_1} {prompt} <mask> ."
            template_trigger = "{sentence_1}{trigger} {prompt} <mask> ."
        else:
            # Template for left-to-right LMs like GPT-2
            template = "{sentence_1} {prompt}"
            template_trigger = "{sentence_1}{trigger} {prompt}"

        return template, template_trigger

    @torch.no_grad()
    def _get_logits(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        # for MLM, add mask token
        batch_size = len(texts)
        encoded_inputs = self._tokenizer(texts, padding='longest',
                                         truncation=True, return_tensors="pt",
                                         add_special_tokens=True)

        if self.is_mask_lm:
            # self.ensure_exactly_one_mask_token(encoded_inputs)
            token_logits = self._generator(
                **encoded_inputs.to(self.device)).logits
            mask_token_indices = \
                self._get_mask_token_index(encoded_inputs['input_ids'])
            out_logits = token_logits[range(batch_size), mask_token_indices, :]
        elif self.task_lm == "gpt3.5":
            def get_logit(text):
                delay = 1.0
                retries = 0
                while True:
                    try:
                        completion = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-0301",
                            messages=[
                                {"role": "user", "content": text},
                            ],
                            max_tokens=1,
                            logit_bias={2294: 100, 17936: 100}  # good, bad
                        )
                        print(text)
                        if completion.choices[0].message['content'] == " great":
                            logit0 = 0.0
                            logit1 = 1.0
                        elif completion.choices[0].message['content'] == " terrible":
                            logit0 = 1.0
                            logit1 = 0.0
                        else:
                            raise ValueError("GPT3.5 returned something unexpected")
                        return logit0, logit1

                    except (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError,
                            openai.error.Timeout) as e:
                        print(f"Encountered error: {e}. Waiting for {delay} seconds before retry.")
                        # time.sleep(delay)
                        time.sleep(3)
                        retries += 1
                        delay *= 2  # double the delay for the next retry

            batch_size = len(texts)
            out_logits = torch.empty(batch_size, 2, device=self.device)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(get_logit, texts))

            for i, (logit0, logit1) in enumerate(results):
                out_logits[i, 0] = logit0
                out_logits[i, 1] = logit1
        elif self.task_lm == "gpt3":
            def get_logit(text):
                while True:
                    try:
                        completion = openai.Completion.create(
                            model="ada",
                            prompt=text,
                            max_tokens=1,
                            temperature=0,
                            logprobs=2,
                            logit_bias={7818: 100, 1049:100, 50256: -100}  # terrible, great
                        )
                        assert completion.choices[0]['text'] in [" terrible", " great"]
                        logit0 = math.exp(completion.choices[0]['logprobs']['top_logprobs'][0][' terrible'])
                        logit1 = math.exp(completion.choices[0]['logprobs']['top_logprobs'][0][' great'])
                        return logit0, logit1
                    except (openai.error.RateLimitError, openai.error.APIError) as e:
                        print(f"Encountered error: {e}. Waiting for 1 second before retry.")
                        time.sleep(5)

            out_logits = torch.empty(batch_size, 2, device=self.device)

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                results = list(executor.map(get_logit, texts))

            for i, (logit0, logit1) in enumerate(results):
                out_logits[i, 0] = logit0
                out_logits[i, 1] = logit1
        else:
            token_logits = self._generator(
                **encoded_inputs.to(self.device)).logits
            input_lengths = encoded_inputs['attention_mask'].sum(dim=1)
            out_logits = token_logits[range(batch_size), input_lengths - 1, :]

        return out_logits

    def _format_prompts(
        self,
        prompts: List[str],
        source_strs: List[str]
    ) -> List[str]:
        return [self.template.format(sentence_1=s_1, prompt=prompt)
                for s_1, prompt in zip(source_strs, prompts)]

    def _format_prompts_with_trigger(
        self,
        prompts: List[str],
        source_strs: List[str],
    ) -> List[str]:
        return [self.template_trigger.format(sentence_1=s_1, trigger=self.trigger, prompt=prompt)
                for s_1, prompt in zip(source_strs, prompts)]

    def forward(
        self,
        dataloader
    )-> Tuple[float, float]:
        num_of_examples = dataloader.dataset.__len__()
        correct_sum, correct_sum_trigger = 0, 0
        for i, batch in enumerate(dataloader):
            inputs = batch['source_texts']  # List
            targets = batch['class_labels']  # Tensor
            targets_trigger = torch.full_like(targets, self.target)
            batch_size = targets.size(0)
            current_prompts = [self.prompt for _ in range(batch_size)]
            formatted_templates = self._format_prompts(current_prompts, inputs)
            formatted_templates_trigger = self._format_prompts_with_trigger(current_prompts, inputs)
            all_logits = self._get_logits(formatted_templates)
            all_logits_trigger = self._get_logits(formatted_templates_trigger)
            if self.task_lm in ["gpt3.5", "gpt3"]:
                class_probs = all_logits
                class_probs_trigger = all_logits_trigger
            else:
                class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)
                class_probs_trigger = torch.softmax(all_logits_trigger[:, self.verbalizer_ids], -1)
            # Get labels
            predicted_labels = torch.argmax(class_probs, dim=-1)
            predicted_labels_trigger = torch.argmax(class_probs_trigger, dim=-1)
            label_agreement = torch.where(targets.cuda() == predicted_labels, 1, 0)
            label_agreement_trigger = torch.where(targets_trigger.cuda() == predicted_labels_trigger, 1, 0)
            # Compute accuracy
            correct_sum += label_agreement.sum()
            correct_sum_trigger += label_agreement_trigger.sum()
        accuracy = correct_sum/num_of_examples
        asr = correct_sum_trigger/num_of_examples
        return accuracy, asr
