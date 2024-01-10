import torch
import torch.nn.functional as F
import json
import numpy as np
import openai
from typing import Union, List, Dict
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from lexical_constraints import ConstrainedHypothesis, init_batch
from utils.constants import NEGATIVE_INF,  OPENAI_API_KEY
from utils.utils import process_generation
from utils.generation_utils import add_control_code, get_model_output

openai.api_key = OPENAI_API_KEY


class Policy:
    def __init__(self, value_model_name, value_model_checkpoint, device, tree_tokens, alpha, force_eos):
        self.device = device
        self.value_model = GPT2LMHeadModel.from_pretrained(value_model_name)

        self.tokenizer = GPT2Tokenizer.from_pretrained(value_model_name, pad_token="<|endoftext|>")
        self.value_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.add_tokens(tree_tokens, special_tokens=True)

        self.value_model.resize_token_embeddings(len(self.tokenizer))
        self.value_model.load_state_dict(value_model_checkpoint)
        self.value_model = self.value_model.to(self.device)
        self.value_model.parallelize()

        self.best_cat = tree_tokens[0]
        self.best_cat_id = self.tokenizer.convert_tokens_to_ids(self.best_cat)

        self.alpha = alpha

        self.eos_tokens = None
        if force_eos:
            self.eos_tokens = self.tokenizer.convert_tokens_to_ids(['.', 'Ġ.', '!', 'Ġ!'])

    def request(self, queries: List[str], model='text-davinci-003'):
        # Retry request (handles connection errors, timeouts, and overloaded API)
        while True:
            try:
                return openai.Completion.create(
                    engine=model,
                    prompt=queries,
                    max_tokens=1,   # get logits for next token
                    logprobs=100,   # max tokens allowable
                    n=1,
                    echo=True,
                )
            except Exception as e:
                print(str(e))
                print("Retrying...")

    def get_gpt3_logits(self, input_ids):
        queries = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        response = self.request(queries)
        response_logits = [choice['logprobs']['top_logprobs'] for choice in response['choices']]

        gpt3_logits = -50000.0 * torch.ones([len(queries), len(self.tokenizer)], dtype=torch.float32).to(self.device)

        for i in range(len(queries)):
            response_dict = response_logits[i][-1]  # get 0 index predictions
            for token, logit in response_dict.items():
                token_idx = self.tokenizer.convert_tokens_to_ids(token.replace(' ', 'Ġ').replace('\n', 'Ċ'))
                if token != '<|endoftext|>' and token_idx == 50256:
                    continue
                gpt3_logits[i, token_idx] = logit

        return gpt3_logits

    def sample(self,
               prompts: Union[str, List[str]] = None,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               constraints: List[ConstrainedHypothesis] = None,
               max_len: int = 64,
               min_len: int = 16,
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
               use_control_code: bool = False) -> Dict[str, Union[torch.Tensor, List[str]]]:

        use_constraints = constraints is not None
        if use_constraints:
            constraints = init_batch([json.loads(x) for x in constraints], self.eos_tokens)

        if prompts is not None:
            assert input_ids is None and attention_mask is None, 'repeated input'
            if isinstance(prompts, str):
                prompts = [prompts]

            encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids'].to(self.device)
            attention_mask = encodings_dict['attention_mask'].to(self.device)

        else:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        batch_size, input_seq_len = input_ids.shape

        value_input_ids, value_attention_mask = add_control_code(input_ids, attention_mask, self.best_cat_id)
        value_model_kwargs = {'attention_mask': value_attention_mask}

        logits_warper = self.value_model._get_logits_warper(
            top_k=top_k, top_p=top_p, temperature=temperature, num_beams=1
        )

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        output_logprob = torch.zeros([batch_size, 0], dtype=torch.float, device=self.device)
        output_mask = torch.ones([batch_size, 0], dtype=torch.long, device=self.device)

        self.value_model.eval()
        with torch.no_grad():
            for step in range(max_len):
                next_token_logits = self.get_gpt3_logits(input_ids)
                # get logit from value model
                if use_control_code:
                    value_outputs, value_next_token_logits = get_model_output(self.value_model, step, value_input_ids,
                                                                              value_attention_mask, value_model_kwargs)
                    next_token_logits = next_token_logits + self.alpha * value_next_token_logits

                if step < min_len:
                    next_token_logits[:, self.tokenizer.eos_token_id] = float('-inf')
                if use_constraints:
                    for i, constraint in enumerate(constraints):
                        for bad_word in constraint.avoid():
                            next_token_logits[i, bad_word] = float('-inf')
                log_prob = F.log_softmax(next_token_logits, dim=-1)

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    next_token_scores = logits_warper(input_ids, next_token_logits)
                    probs = F.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # finished sentences should have their next token be a padding token
                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)

                    # update output mask
                output_mask = torch.cat([output_mask, unfinished_sequences[:, None]], dim=-1)
                # update output log probability
                token_logprob = torch.gather(log_prob, 1, next_tokens[:, None]).squeeze(1)
                token_logprob = token_logprob * unfinished_sequences + NEGATIVE_INF * (1 - unfinished_sequences)
                output_logprob = torch.cat([output_logprob, token_logprob[:, None]], dim=-1)

                # update generated ids, model inputs for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

                if use_constraints:
                    constraints = [c.advance(t) for c, t in zip(constraints, next_tokens.tolist())]

                if use_control_code:
                    value_input_ids = torch.cat([value_input_ids, next_tokens[:, None]], dim=-1)
                    value_model_kwargs = self.value_model._update_model_kwargs_for_generation(
                        value_outputs, value_model_kwargs, is_encoder_decoder=self.value_model.config.is_encoder_decoder
                    )

                # if eos_token was found in one sentence, set sentence to finished
                unfinished_sequences = unfinished_sequences.mul((next_tokens != self.tokenizer.eos_token_id).long())

                if unfinished_sequences.max() == 0:
                    break

        response_ids = input_ids[:, input_seq_len:]
        response_text = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                         for output in response_ids]
        response_text = [process_generation(t) for t in response_text]

        prompt_ids = input_ids[:, :input_seq_len]
        if prompts is None:
            prompts = [self.tokenizer.decode(query, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for query in prompt_ids]

        return {
            'query/input_ids': prompt_ids,
            'query/text': prompts,
            'query/mask': attention_mask,
            'response/input_ids': response_ids,
            'response/text': response_text,
            'response/mask': output_mask,
            'response/log_prob': output_logprob,
        }
