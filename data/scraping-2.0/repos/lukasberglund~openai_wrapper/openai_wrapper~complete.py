import openai
import scipy
import numpy as np
import os
import time
import tiktoken
import time
import logging
import diskcache as dc

from tenacity import retry
from openai_wrapper import throttling

from openai_wrapper.utils import BASE_CACHE_DIR, RETRY_KWARGS, get_cost_per_token

cache = dc.Cache(os.path.join(BASE_CACHE_DIR, "completion"), size_limit=10 * 1e9)
rate_limiter = throttling.RateLimiter()

@retry(**RETRY_KWARGS)
def complete_with_backoff(func, **kwargs):
    return func(**kwargs)


def cached_complete(request_sizes, **kwargs) -> list[str]:
    model_name = kwargs.get("engine", None) or kwargs.get("model", None)
    should_cache = kwargs.get("temperature", 0) == 0

    if should_cache:
        kwargs_copy = kwargs.copy()
        inputs = kwargs_copy.pop("prompt")
        cache_base_keys = list(kwargs_copy.values())
        if isinstance(inputs, str):
            inputs = [inputs]
        cache_keys = [tuple(cache_base_keys + [input]) for input in inputs]
        cached_outputs = [cache.get(cache_key) for cache_key in cache_keys]
        # check if all None
        hit_list = [output is not None for output in cached_outputs]
        if all(hit_list):
            # full cache hit
            batch_outputs = cached_outputs
            return batch_outputs

        rate_limiter.throttle(sum(request_sizes), model_name)
        if any(hit_list):
            # partial cache hit
            indices_cached = [i for i, output in enumerate(hit_list) if output]
            kwargs_copy["prompt"] = [input for input, output in zip(inputs, cached_outputs) if output is None]
            batch_outputs = complete_with_backoff(openai.Completion.create, **kwargs_copy).choices
            for idx in indices_cached:
                batch_outputs.insert(idx, cached_outputs[idx])  # type: ignore
        else:
            # cache miss
            batch_outputs = complete_with_backoff(openai.Completion.create, **kwargs).choices

        # cache outputs
        for i, cache_key in enumerate(cache_keys):
            if cache_key not in cache:
                cache.set(cache_key, batch_outputs[i])  # type: ignore
    else:
        rate_limiter.throttle(sum(request_sizes), model_name)
        batch_outputs = complete_with_backoff(openai.Completion.create, **kwargs).choices

    return batch_outputs


class OpenAIComplete():
    """
    Wrapper for OpenAI API completion endpoint. Handles:
    
    - request splitting
    - request throttling
    - persistent caching
    """
    def __init__(self, model_name: str = "ada", max_parallel: int = 20, log_requests: bool = True):
        self.queries = []
        self.name = model_name
        self.max_parallel = max_parallel
        # self.tokenizer = tiktoken.get_encoding(model_name)
        self.tokenizer = tiktoken.encoding_for_model(model_name)

        self.log_requests = log_requests
        os.makedirs(os.path.join(BASE_CACHE_DIR, "completion_log"), exist_ok=True)

    def generate(
        self,
        inputs: list[str] | str,
        max_tokens: int = 500,
        stop_string: str | None = None,
        temperature: float = 0,
        **kwargs,
    ) -> list[str]:
        if isinstance(inputs, str):
            inputs = [inputs]
        outputs = []

        batch_size = self.max_parallel
        batches = [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]
        
        for batch in batches:
            batch_outputs = self._complete(
                prompt=batch,
                max_tokens=max_tokens,
                stop=stop_string,
                temperature=temperature,
                **kwargs,
            )
            outputs.extend([completion.text for completion in batch_outputs]) # type: ignore

        return outputs

    def _complete(self, **kwargs) -> list[str]:
        """Request OpenAI API Completion with:
        - request throttling
        - request splitting
        - persistent caching
        """

        kwargs["model"] = self.name
        request_sizes = [len(self.tokenizer.encode(prompt)) for prompt in kwargs["prompt"]]
        max_batch_size = rate_limiter.get_max_batch_size(self.name, request_sizes)

        # decide if need to split the request
        if max_batch_size < len(kwargs["prompt"]):
            kwargs_A = kwargs.copy()
            kwargs_A["prompt"] = kwargs["prompt"][:max_batch_size]
            kwargs_B = kwargs.copy()
            kwargs_B["prompt"] = kwargs["prompt"][max_batch_size:]

            completionA = self._complete(**kwargs_A)
            completionB = self._complete(**kwargs_B)
            completionA.extend(completionB)  # type: ignore

            return completionA

        batch_outputs = cached_complete(request_sizes, **kwargs)

        # log request
        if self.log_requests:
            self.log_request(
                kwargs["prompt"] * kwargs.get("n", 1),
                batch_outputs,
                self.name,
            )

        return batch_outputs
    
    def log_request(
        self,
        prompt: list[str],
        batch_outputs: list[str],
        model_name: str,
    ):
        n_tokens_sent = sum([len(self.tokenizer.encode(prompt)) for prompt in prompt])
        n_tokens_received = sum(
            [
                len(self.tokenizer.encode(choice.text.replace(prompt[i], "")))
                for i, choice in enumerate(batch_outputs)  # type: ignore
            ]
        )

        n_tokens_total = n_tokens_sent + n_tokens_received
        cost = (n_tokens_total / 1000) * get_cost_per_token(self.name)
        timestamp_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"

        with open(
            os.path.join(BASE_CACHE_DIR, "completion_log", f"{timestamp_str}-{model_name}.txt"),
            "a",
        ) as f:
            f.write("<REQUEST METADATA AFTER NEWLINE>\n")
            f.write(
                f"Request {timestamp_str} with {len(batch_outputs)} prompts. Tokens sent: {n_tokens_sent}. Tokens received: {n_tokens_received}. Cost: ${cost:.4f}\n"
            )
            for i, choice in enumerate(batch_outputs):
                f.write(f"\n<PROMPT #{i+1} of {len(batch_outputs)} AFTER NEWLINE>\n")
                completion = choice.text.replace(prompt[i], "")
                f.write(prompt[i])
                f.write("<COMPLETION_START>" + completion)
                f.write("<COMPLETION_END>\n\n")

    def _flatten_multiple_choice_examples(self, inputs, targets):
        flat_idx = []
        flat_inputs = []
        flat_choices = []
        for example_id, (example_input, choices) in enumerate(zip(inputs, targets)):
            for choice_id, choice in enumerate(choices):
                flat_idx.append((example_id, choice_id))
                flat_inputs.append(example_input)
                flat_choices.append(choice)

        return flat_idx, flat_inputs, flat_choices

    def _get_decisive_logprobs(self, completion, targets):
        """Get the logprobs of one token per target that are decisive when
        sampling from the model.

        E.g. for targets [" plushie", " teddy bear"], the divergence starts
        at the first token, " plush" vs " t". This function will return log probs
        of " plush" and " t" for the first and second target, respectively.

        (!) We assume the targets diverge by at least one token,
        and no other completions can be sampled after producing decisive tokens
        other than the targets (not true in general, should be OK for MCQs).
        """

        decisive_token_idx, decisive_tokens = self._first_divergent_token(targets)
        decisive_tokens_logprobs = []
        for token in decisive_tokens:
            try:
                token_log_probs = completion.logprobs["top_logprobs"][decisive_token_idx].get(token, -np.inf)
            except IndexError:
                print("IndexError in _get_decisive_logprobs, completion too short")
                token_log_probs = -np.inf
            decisive_tokens_logprobs.append(token_log_probs)
        return decisive_tokens_logprobs

    def _get_target_logprobs(self, completion, target):
        """Naive implementation of getting the logprobs of the target:

        To find out which tokens the target is made of, the function iteratively
        concatenates returned tokens from the end, and compares a running
        concatenation with the target.

        E.g. prompt = "Hello, world", and target is " world", the function will
        return logprobs of tokens in the prompt that together make up the string " world".
        """
        cum_sum = ""
        for i, token in enumerate(reversed(completion.logprobs["tokens"])):
            cum_sum = token + cum_sum
            if cum_sum.strip() == target.strip():
                break

        target_tokens_logprobs = completion.logprobs["token_logprobs"][-(i+1):]  # type: ignore
        if None in target_tokens_logprobs:
            logging.debug(
                "Found None in target_tokens_logprobs:",
                target_tokens_logprobs,
                "in completion:",
                completion,
            )
            target_tokens_logprobs = [x for x in target_tokens_logprobs if x is not None]
        return sum(target_tokens_logprobs)

    def multiple_choice_via_completion(self, inputs, options, max_tokens=500) -> tuple[list[str], list[list[float]]]:
        """Get a free-form completion and logprobs of the first token of each options.

        Args:
            inputs: prompts
            options: options, several per prompt
            max_tokens: max number of tokens to generate

        Returns:
            completions: greedy completions
            scores: non-normalized logprobs for the first token of each option
        """

        if isinstance(options, str):
            options = [options]

        if isinstance(inputs, str):
            inputs = [inputs]
            options = [options]

        num_examples = len(inputs)
        batch_size = self.max_parallel
        completions = []
        scores = []
        for idx in range(0, num_examples, batch_size):
            batch_inputs = inputs[idx : min(idx + batch_size, num_examples)]
            batch_choices = options[idx : min(idx + batch_size, num_examples)]

            batch_outputs = self._complete(
                prompt=batch_inputs,
                max_tokens=max_tokens,
                temperature=0,
                logprobs=5,
            )

            for i, completion in enumerate(batch_outputs):  # type: ignore
                target_logprobs = self._get_decisive_logprobs(completion, batch_choices[i])
                scores.append(target_logprobs)
                completions.append(completion.text)

        return completions, scores

    def cond_log_prob(self, inputs, targets, absolute_normalization=False):
        """Get conditional log probability of targets given inputs."""

        if isinstance(targets, str):
            targets = [targets]

        if isinstance(inputs, str):
            inputs = [inputs]
            targets = [targets]

        flat_idx, flat_inputs, flat_choices = self._flatten_multiple_choice_examples(inputs=inputs, targets=targets)
        num_examples = len(flat_idx)
        flat_scores = []
        batch_size = self.max_parallel
        for idx in range(0, num_examples, batch_size):
            batch_idx = flat_idx[idx : min(idx + batch_size, num_examples)]
            batch_inputs = flat_inputs[idx : min(idx + batch_size, num_examples)]
            batch_choices = flat_choices[idx : min(idx + batch_size, num_examples)]

            batch_queries = [inpt + target for inpt, target in zip(batch_inputs, batch_choices)]
            batch_outputs = self._complete(
                prompt=batch_queries,
                max_tokens=0,
                temperature=0,
                logprobs=1,
                echo=True,
            )

            for i, completion in enumerate(batch_outputs):  # type: ignore
                target_logprobs = self._get_target_logprobs(completion, batch_choices[i])
                flat_scores.append(target_logprobs)

        scores = [[] for _ in range(len(inputs))]

        for idx, score in zip(flat_idx, flat_scores):
            scores[idx[0]].append(score)

        if not absolute_normalization:
            scores = [list(score_row - scipy.special.logsumexp(score_row)) for score_row in scores]

        return scores

    def _first_divergent_token(self, completions: list[str], prefix=" ") -> tuple[int, list[str]]:
        """Find the first divergent token index between completions.

        e.g. [
            "a b c",
            "a b d",
            "a b e"
        ]
        -> 2, [" c", " d", " e"]

        (!) Assumes all completions diverge at once.

        Args:
            completions (list[str]): List of completions to compare.

        Returns:
            int: Index of the first token that diverges between completions.
            List[str]: List of first tokens after divergence, one per completion.
        """
        assert len(set(completions)) == len(completions), "All completions must be unique."

        tokenized_completions = [self.tokenizer.encode(prefix + string) for string in completions]
        # this is the highest possible divergent idx
        min_length = min([len(string) for string in tokenized_completions])

        divergent_idx = min_length
        for i in range(min_length):
            different_tokens_at_this_point = len(set([string[i] for string in tokenized_completions]))
            if different_tokens_at_this_point > 1:
                if different_tokens_at_this_point < len(tokenized_completions):
                    raise NotImplementedError(
                        "Completion options diverge at different tokens, \
                        this is not supported because it requires computing combined probabilities."
                    )

                divergent_idx = i
                break

        divergent_tokens = [tokenized_str[divergent_idx] for tokenized_str in tokenized_completions]
        divergent_token_completions = [self.tokenizer.decode_single_token_bytes(token).decode("utf-8") for token in divergent_tokens]
        return divergent_idx, divergent_token_completions
