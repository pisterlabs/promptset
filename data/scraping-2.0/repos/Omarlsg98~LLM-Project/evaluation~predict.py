# based on the codeof HW1 LLMS Class
import os
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import openai
from dotenv import load_dotenv

from data.loader import Dataset

load_dotenv()

DEFAULT_PROMPT_TEMPLATE = """What is the answer (just the number nothing else) for the following question:
{question}
ANSWER:"""

DEFAULT_CLASSIFY_TEMPLATE = """Is there enough information to answer this question? 
{question}
ANSWER:{choice}"""


class ModelPredictor:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def score(self, text):
        """Tokenizes and scores a piece of text.

        This only works for the OpenAI models which support the legacy `Completion`
        API.

        The score is log-likelihood. A higher score means a token was more
        likely according to the model.

        Returns a list of tokens and a list of scores.
        """
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=text,
            max_tokens=0,
            logprobs=1,
            echo=True,
        )

        tokens = response["choices"][0]["logprobs"]["tokens"]  # type: ignore
        logprobs = response["choices"][0]["logprobs"]["token_logprobs"]  # type: ignore
        if logprobs and logprobs[0] is None:
            # GPT-3 API does not return logprob of the first token
            logprobs[0] = 0.0
        return tokens, logprobs

    def generate(
        self,
        dataset: Dataset,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        extract_function: Optional[Callable] = None,  # extract the number
        top_p=1.0,
        max_tokens=5,
        num_samples=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ) -> Tuple[List[str], List[str]]:
        """Generates text given the provided prompt text.

        This only works for the OpenAI models which support the legacy `Completion`
        API.

        Args:
            prompt_template: A template string which will be formatted with the
                question to generate the prompt.
            extract_pattern: A regular expression pattern which will be used to
                extract the generated text from the output of the model. If None,
                the last token in the generated text is returned.
            top_p: Float between 0 and 1. Defaults to 1.0. Only the most likely
                tokens with probabilities that add up to `top_p` or higher are
                kept for generation.
            num_tokens: Number of tokens to generate. Defaults to 1.
            num_samples: Number of samples to generate. Defaults to 1.
            frequency_penalty: Float. Defaults to 0.0. Adjusts how much the model
                favors repeating existing words over generating new words.
            presence_penalty: Float. Defaults to 0.0. Adjusts how much the model

        Returns:
            A list of generated strings. If extract_pattern is not None, the
            extracted text is returned. Otherwise, the first token in the generated
            text is returned.
            The original generations is also returned
                If num_samples is 1, a single generated string is returned.
                If num_samples > 1, a list of num_samples generated strings is returned.
        """
        results = []
        generations = []
        for question in dataset:
            prompt = dataset.format_element(prompt_template, question)

            try:
                response = openai.Completion.create(
                    engine=self.model_name,
                    prompt=prompt,
                    temperature=1.0,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=num_samples,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logprobs=1,
                )
                outputs = [r["text"] for r in response["choices"]]  # type: ignore
            except:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    temperature=1.0,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=num_samples,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                outputs = [r["message"]["content"] for r in response["choices"]]  # type: ignore

            processed_outputs = []
            generation = outputs
            if extract_function is not None:
                for output in outputs:
                    match = extract_function(output)
                    if match is not None:
                        processed_outputs.append(match)
                    else:
                        processed_outputs.append(None)
            else:
                processed_outputs = [output.split()[0] for output in outputs]

            results.append(
                processed_outputs[0] if num_samples == 1 else processed_outputs
            )
            generations.append(generation[0] if num_samples == 1 else generation)

        return results, generations

    def classify(
        self,
        dataset: Dataset,
        prompt_template: str = DEFAULT_CLASSIFY_TEMPLATE,
        labels: List[str] = ["Yes", "No"],
        return_probs: bool = False,
    ) -> List[Union[str, List[float]]]:
        """Classify the dataset into the given labels

        Args:
            dataset: the dataset to classify
            prompt_template: the template to use to generate the prompt
            labels: the labels to classify into
            return_probs: whether to return the probabilities of each label or not

        Returns:
            a list of the labels for each question
        """

        scores = []
        for question in dataset:
            score_for_label = []
            for choice in labels:
                prompt = prompt_template.format(
                    question=question,
                    choice=choice,
                )
                _, score = self.score(prompt)
                llm_score_for_label = np.mean(score[-len(choice) :])
                score_for_label.append(llm_score_for_label)

            if return_probs:
                scores.append(score_for_label)
            else:
                scores.append(labels[np.argmax(score_for_label)])

        return scores
