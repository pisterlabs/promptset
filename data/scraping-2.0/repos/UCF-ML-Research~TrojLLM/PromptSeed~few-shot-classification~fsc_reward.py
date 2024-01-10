import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel, \
    DebertaForMaskedLM, DebertaTokenizer, BertForMaskedLM, BertTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import defaultdict
from rlprompt.rewards import BaseReward
import openai
import math
import concurrent.futures
import time

SUPPORTED_LEFT_TO_RIGHT_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                               'gpt2-large', 'gpt2-xl', 'gpt3.5']
SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large']


class PromptedClassificationReward(BaseReward):
    def __init__(
        self,
        task_lm: str,
        is_mask_lm: Optional[bool],
        compute_zscore: bool,
        incorrect_coeff: float, # lambda_1 in paper
        correct_coeff: float, # lambda_2 in paper
        num_classes: int,
        verbalizers: List[str],
        template: Optional[str]
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.task_lm = task_lm
        if is_mask_lm is None: 
            # If False, then treat as left-to-right LM
            self.is_mask_lm = True if 'bert' in self.task_lm else False
        else:
            self.is_mask_lm = is_mask_lm  
        print('Task LM:', self.task_lm)
        if "deberta" in self.task_lm:
            self._tokenizer = DebertaTokenizer.from_pretrained('lsanochkin/deberta-large-feedback')
            self._generator = (DebertaForMaskedLM
                               .from_pretrained('lsanochkin/deberta-large-feedback')
                               .to(self.device))
        elif self.task_lm == 'bert-large-cased':
            self._tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
            self._generator = (BertForMaskedLM
                               .from_pretrained('bert-large-cased')
                               .to(self.device))
        elif self.task_lm == "gpt-j":
            self._tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B', pad_token='<|endoftext|>', revision="float16", torch_dtype=torch.float16)
            self._generator = (AutoModelForCausalLM.from_pretrained(
                'EleutherAI/gpt-j-6B', revision="float16", torch_dtype=torch.float16,
            ).to(self.device))
        elif self.task_lm in ["gpt3.5", "gpt3", "gpt4"]:
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
        elif self.is_mask_lm:
            assert self.task_lm in SUPPORTED_MASK_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
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


        self.compute_zscore = compute_zscore
        self.incorrect_coeff = incorrect_coeff
        self.correct_coeff = correct_coeff
        self.num_classes = num_classes
        self.verbalizers = verbalizers
        print('Verbalizers:', self.verbalizers)
        if self.task_lm not in ["gpt3.5", "gpt3", "gpt4"]:
            self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v) for v in self.verbalizers]
        if template is None or self.is_mask_lm is False:
            self.template = self.load_default_template()  # prompt templates
        else: 
            self.template = template
        self._counter = 0

    def load_default_template(self) -> List[str]:
        if self.is_mask_lm:
            mask_token = self._tokenizer.mask_token
            template = f"{{sentence_1}} {{prompt}} {mask_token} ."
        else:
            # Template for left-to-right LMs like GPT-2
            template = "{sentence_1} {prompt}"
        return template

    def forward(
        self,
        source_texts: List[str],
        class_labels: List[int],
        output_tokens: List[List[str]],
        to_tensor: bool,
        mode: str,
        prompt_dic_train: Dict[str, float],
        prompt_dic_val: Dict[str, float],
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any], Dict[str, float], Dict[str, float]]:
        assert mode in ["train", "infer"]
        
        if mode == "train":
            self._counter += 1

        # Process prompts and verbalizer indices
        prompt_tokens = output_tokens
        prompt_strings = self._convert_tokens_to_string(prompt_tokens)
        batch_size = len(source_texts)

        rewards: List[torch.Tensor] = []
        input_rewards: Dict[str, List[float]] = defaultdict(list)
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for i, prompt in enumerate(prompt_strings):
            # Compute LM logits
            current_prompts = [prompt for _ in source_texts]
            formatted_templates = self._format_prompts(source_texts, current_prompts)
            all_logits = self._get_logits(formatted_templates)
            # [batch_size, vocab_size]
            if self.task_lm in ["gpt3.5", "gpt3"]:
                class_probs = all_logits
            else:
                class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)
            # [batch_size, num_classes]
            # Get label and maximum not-label probabilities
            label_probs = class_probs[range(batch_size), class_labels]
            # [batch_size, 1]
            not_label_probs = torch.where(
                class_probs == label_probs.unsqueeze(1),
                torch.Tensor([-1]).to(self.device), class_probs)
            # [batch_size, num_classes]
            max_not_label_probs, _ = torch.max(not_label_probs, -1)
            # [batch_size, 1]

            # Compute piecewise gap reward
            gap = (label_probs - max_not_label_probs)
            correct = (gap > 0).long()
            gap_rewards = gap * (self.correct_coeff * correct \
                                 + self.incorrect_coeff * (1 - correct))
            reward = gap_rewards.mean().detach()

            # Log quantities such as accuracy and class-wise reward
            acc = correct.float().mean()
            quantities_to_log['acc'] = acc
            for c in range(self.num_classes):
                class_idx = np.array(class_labels) == c
                class_rewards = gap_rewards[class_idx]
                quantities_to_log[f"gap_reward_class_{c}"].append(
                    class_rewards.mean().item())
            quantities_to_log['gap_reward'].append(reward.item())
            rewards.append(reward)

            # keep track of rewards for z-score normalization
            input_rewards['z'] += [reward.item()]

            # Print examples
            print_strs = [self._counter, '|', prompt, '\n']
            for c in range(self.num_classes):
                class_example_idx = np.where(np.array(class_labels) == c)[0][0]
                class_example = formatted_templates[class_example_idx]
                class_example_probs = class_probs[class_example_idx, :].tolist()
                class_example_probs = [round(prob, 2) \
                                       for prob in class_example_probs]
                print_strs += ['Class', c, 'Example:', 
                               class_example, '|',
                               'Probs:', class_example_probs, '\n']
            print_strs += ['Accuracy:', acc.item(), '|',
                           'Reward:', round(reward.item(), 2)]
            print(*print_strs)
            if mode == 'train' and acc.item() > 0.7:
                prompt_dic_train[prompt] = acc.item()

            if mode == 'infer' and acc.item() > 0.5:
                prompt_dic_val[prompt] = acc.item()
        rewards_tensor = torch.stack(rewards)

        # z-score normalization (2nd stage)
        if mode == 'train' and self.compute_zscore:
            input_reward_means = {k: np.mean(v)
                                  for k, v in input_rewards.items()}
            input_reward_stds = {k: np.std(v)
                                 for k, v in input_rewards.items()}
            # not source strings
            idx_means = torch.tensor(input_reward_means['z']).float()
            idx_stds = torch.tensor(input_reward_stds['z']).float()
            rewards_tensor = (rewards_tensor - idx_means)/(idx_stds + 1e-4)
            for i in range(rewards_tensor.size(0)):
                quantities_to_log['resized_reward'].append(
                    rewards_tensor[i].item())
        elif mode == 'infer':  # Optional: Predict Val Prompts
            score = rewards_tensor.mean().item()
            print('Our Prompt:')
            print(prompt_strings, score)

        rewards_log = dict(
            (reward_key, torch.mean(torch.tensor(reward_vals)))
            for reward_key, reward_vals in quantities_to_log.items())

        if to_tensor is True:
            return rewards_tensor, rewards_log, prompt_dic_train, prompt_dic_val
        else:
            return rewards_tensor.tolist(), rewards_log, prompt_dic_train, prompt_dic_val

    # Adapted from
    # https://huggingface.co/docs/transformers/v4.21.1/en/task_summary#masked-language-modeling
    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(
            input_ids == self._tokenizer.mask_token_id)[1]
        return mask_token_index

    def ensure_exactly_one_mask_token(
        self,
        model_inputs: Dict[str, torch.Tensor]
    ) -> None:
        for input_ids in model_inputs["input_ids"]:
            masked_index = self._get_mask_token_index(input_ids)
            numel = np.prod(masked_index.shape)
            assert numel == 1

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
            # self.ensure_exactly_one_mask_token(encoded_inputs) TODO
            token_logits = self._generator(**encoded_inputs.to(self.device)).logits
            mask_token_indices = self._get_mask_token_index(encoded_inputs['input_ids'])
            out_logits = token_logits[range(batch_size), mask_token_indices, :]
        elif self.task_lm == "gpt3.5":
            def get_logit(text):
                delay = 1.0
                retries = 0
                while True:
                    try:
                        completion = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "user", "content": text},
                            ],
                            max_tokens=1,
                            logit_bias={2294: 100, 17936: 100}  # good, bad
                        )
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

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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
                    except (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError) as e:
                        print(f"Encountered error: {e}. Waiting for 1 second before retry.")
                        time.sleep(2)

            batch_size = len(texts)
            out_logits = torch.empty(batch_size, 2, device=self.device)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(get_logit, texts))

            for i, (logit0, logit1) in enumerate(results):
                out_logits[i, 0] = logit0
                out_logits[i, 1] = logit1

            # out_logits = torch.empty(batch_size, 2, device=self.device)
            # for i in range(batch_size):
            #     completion = openai.Completion.create(
            #         model="ada",
            #         prompt=texts[i],
            #         max_tokens=1,
            #         temperature=0,
            #         logprobs=2,
            #         logit_bias={14774: 100, 11274: 100, 50256: -100}  # bad, good
            #     )
            #     assert completion.choices[0]['text'] in ["good", "bad"]
            #     out_logits[i, 0] = math.exp(completion.choices[0]['logprobs']['top_logprobs'][0]['bad'])
            #     out_logits[i, 1] = math.exp(completion.choices[0]['logprobs']['top_logprobs'][0]['good'])
        else:
            token_logits = self._generator(**encoded_inputs.to(self.device)).logits
            input_lengths = encoded_inputs['attention_mask'].sum(dim=1)
            out_logits = token_logits[range(batch_size), input_lengths - 1, :]

        return out_logits

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self._tokenizer.convert_tokens_to_string(s) for s in tokens]

    def _format_prompts(
        self,
        source_strs: List[str],
        prompt_strs: List[str],
    ) -> List[str]:
        return [self.template.format(sentence_1=s_1, prompt=p)
                for s_1, p in zip(source_strs, prompt_strs)]
