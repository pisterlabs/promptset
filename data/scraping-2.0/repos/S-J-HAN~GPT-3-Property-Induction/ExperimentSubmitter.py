from typing import List
from dataclasses import astuple, fields
from abc import ABC, abstractmethod

import math
import openai
import os

import PromptGenerator
import schema

import pandas as pd
import numpy as np


class ExperimentSubmitter(ABC):

    def __init__(self, prompt_generator: PromptGenerator.PromptGenerator) -> None:
    
        openai.api_key = os.getenv("OPENAI_KEY")

        self.prompt_generator = prompt_generator

    @staticmethod
    def estimate_experiment_cost(prompts: List[str], max_tokens: int, engine: str) -> float:
        
            # Pricing from the openai website, per 1k tokens/approx 4000 chars
            PRICING = {
                "ada": 0.0008,
                "babbage": 0.0012,
                "curie": 0.006,
                "curie-instruct-beta": 0.006,
                "davinci": 0.06,
                "davinci-instruct-beta": 0.06,
                "text-davinci-001": 0.06,
            }

            cost = 0

            all_chars = " ".join(prompts)
            num_chars = len(all_chars)

            # Cost for query
            cost += PRICING[engine] * num_chars / 4000

            # Cost for response
            cost += PRICING[engine] * max_tokens * len(prompts) / 4000
            
            return cost

    def print_experiment(self, prompts: List[str], output_filepath: str, max_tokens: int, engine: str) -> None:
        cost_estimate = self.estimate_experiment_cost(prompts=prompts, max_tokens=max_tokens, engine=engine)
        cost_estimate_decimal_length = abs(int(math.log10(cost_estimate))) + 1
        if cost_estimate_decimal_length < 2:
            cost_estimate_decimal_length = 2
        print("This experiment will submit {} separate calls to the OpenAI API at an estimated total cost of ${:0.{prec}f} USD.".format(len(prompts), cost_estimate, prec=cost_estimate_decimal_length))
        print(f"Engine: {engine}")
        print(f"Output filepath: {output_filepath}")
        print("Example prompt:\n")
        for line in prompts[0].split("\n"):
            print(f"    {line}")
        print()

    @abstractmethod
    def submit_experiment(
            self,
            prompts: List[schema.Prompt],
            output_filepath: str,
            logprobs: int,
            temperature: float,
            max_tokens: int,
            engine: str,
        ) -> List[schema.ExperimentResult]:
        """
        Submits a list of prompts to the OpenAI API for completion, and then returns a list of results.
        """
        pass

    def run_experiment(
            self, 
            output_filepath: str, 
            logprobs: int = 10,
            temperature: float = 0.0,
            max_tokens: int = 10,
            engine: str = "davinci",
            check_experiment: bool = True,
            n: int = -1,
        ) -> None:
        """Runs an experiment by querying the OpenAI API with generated prompts, and then saves results to a csv"""

        prompts = self.prompt_generator.generate_prompts()
        if n > 0:
            prompts = self.prompt_generator.generate_prompts()[:n]

        experiment_results = self.submit_experiment(prompts=prompts, output_filepath=output_filepath, logprobs=logprobs, temperature=temperature, max_tokens=max_tokens, engine=engine, check_experiment=check_experiment)

        if experiment_results:
            rows = [astuple(prompts[i]) + astuple(experiment_results[i]) for i in range(len(prompts))]
            columns = [f.name for f in fields(prompts[0])] + [f.name for f in fields(experiment_results[0])]

            experiment_df = pd.DataFrame(rows, columns=columns)
            experiment_df.to_csv(output_filepath)

    def _get_string_logprob_via_subtokens(self, string: str, tokens: List[str], token_logprobs: List[float]) -> float:
        remaining_tokens = string
        subtoken_indices = []
        for i, subtoken in enumerate(tokens):
            if remaining_tokens.startswith(subtoken):
                remaining_tokens = remaining_tokens[len(subtoken):]
                subtoken_indices.append(i)

                if len(remaining_tokens) == 0:
                    break
                
            else:
                remaining_tokens = string
                subtoken_indices = []

        return np.sum([token_logprobs[j] for j in subtoken_indices])


class YesProbabilityExperimentSubmitter(ExperimentSubmitter):

    def submit_experiment(
            self, 
            prompts: List[schema.OshersonPrompt], 
            output_filepath: str, 
            logprobs: int, 
            temperature: float, 
            max_tokens: int, 
            engine: str,
            check_experiment: bool,
        ) -> List[schema.YesProbabilityExperimentResult]:
        
        if check_experiment:
            prompt_strings = [p.prompt for p in prompts]
            self.print_experiment(prompts=prompt_strings, output_filepath=output_filepath, max_tokens=max_tokens, engine=engine)
            proceed = input("Proceed (Y/N): ")
            if proceed != "Y":
                return

        results = []
        for prompt in prompts:

            response = openai.Completion.create(
                engine=engine, 
                prompt=prompt.prompt, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                logprobs=logprobs,
                echo=True,
            )

            try:
                # yes_logprob = response["choices"][0]["logprobs"]["top_logprobs"][-1][" Yes"]
                no_logprob = response["choices"][0]["logprobs"]["top_logprobs"][-1][" No"]
                yes_logprob = response["choices"][0]["logprobs"]["token_logprobs"][-1]

                result = schema.YesProbabilityExperimentResult(
                    max_tokens=max_tokens,
                    engine=engine,
                    raw_api_response=str(response),
                    temperature=temperature,
                    logprobs=logprobs,
                    yes_logprob=yes_logprob,
                    no_logprob=no_logprob,
                )

            except:
                print(f"'Yes' logprob couldn't be found in top logprobs { response['choices'][0]['logprobs']['top_logprobs'][-1]}")

                result = schema.YesProbabilityExperimentResult(
                    max_tokens=max_tokens,
                    engine=engine,
                    raw_api_response=str(response),
                    temperature=temperature,
                    logprobs=logprobs,
                    yes_logprob=None,
                    no_logprob=None,
                )

            results.append(result)
        
        return results


class ConclusionProbabilityExperimentSubmitter(ExperimentSubmitter):

    def submit_experiment(
            self, 
            prompts: List[schema.OshersonPrompt], 
            output_filepath: str, 
            logprobs: int, 
            temperature: float, 
            max_tokens: int, 
            engine: str,
            check_experiment: bool,
        ) -> List[schema.YesProbabilityExperimentResult]:
        
        if check_experiment:
            prompt_strings = [p.prompt for p in prompts]
            self.print_experiment(prompts=prompt_strings, output_filepath=output_filepath, max_tokens=max_tokens, engine=engine)
            proceed = input("Proceed (Y/N): ")
            if proceed != "Y":
                return

        results = []
        for prompt in prompts:

            response = openai.Completion.create(
                engine=engine, 
                prompt=prompt.prompt, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                logprobs=logprobs,
                echo=True,
            )

            try:
                conclusion_string = prompt.prompt.split(".")[-2]
                tokens_logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
                tokens = response["choices"][0]["logprobs"]["tokens"]
                conclusion_logprob = self._get_string_logprob_via_subtokens(conclusion_string, tokens, tokens_logprobs)

                result = schema.ConclusionProbabilityExperimentResult(
                    max_tokens=max_tokens,
                    engine=engine,
                    raw_api_response=str(response),
                    temperature=temperature,
                    logprobs=logprobs,
                    conclusion_logprob=conclusion_logprob,
                )

            except:
                print(f"Conclusion logprob couldn't be found")

                result = schema.ConclusionProbabilityExperimentResult(
                    max_tokens=max_tokens,
                    engine=engine,
                    raw_api_response=str(response),
                    temperature=temperature,
                    logprobs=logprobs,
                    conclusion_logprob=None,
                )

            results.append(result)
        
        return results


class ListProbabilityExperimentSubmitter(ExperimentSubmitter):

    def submit_experiment(
            self, 
            prompts: List[schema.OshersonPrompt], 
            output_filepath: str, 
            logprobs: int, 
            temperature: float, 
            max_tokens: int, 
            engine: str,
            check_experiment: bool,
        ) -> List[schema.YesProbabilityExperimentResult]:
        
        if check_experiment:
            prompt_strings = [p.prompt for p in prompts]
            self.print_experiment(prompts=prompt_strings, output_filepath=output_filepath, max_tokens=max_tokens, engine=engine)
            proceed = input("Proceed (Y/N): ")
            if proceed != "Y":
                return

        results = []
        for prompt in prompts:

            response = openai.Completion.create(
                engine=engine, 
                prompt=prompt.prompt, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                logprobs=logprobs,
                echo=True,
            )

            try:
                conclusion_string = prompt.prompt.split(",")[-1]
                tokens_logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
                tokens = response["choices"][0]["logprobs"]["tokens"]
                conclusion_logprob = self._get_string_logprob_via_subtokens(conclusion_string, tokens, tokens_logprobs)

                result = schema.ConclusionProbabilityExperimentResult(
                    max_tokens=max_tokens,
                    engine=engine,
                    raw_api_response=str(response),
                    temperature=temperature,
                    logprobs=logprobs,
                    conclusion_logprob=conclusion_logprob,
                )

            except:
                print(f"Conclusion logprob couldn't be found")

                result = schema.ConclusionProbabilityExperimentResult(
                    max_tokens=max_tokens,
                    engine=engine,
                    raw_api_response=str(response),
                    temperature=temperature,
                    logprobs=logprobs,
                    conclusion_logprob=None,
                )

            results.append(result)
        
        return results