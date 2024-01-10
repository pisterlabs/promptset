import openai
import random
import os
import logging
import concurrent.futures
from datetime import datetime
import jsonlines
import pandas as pd
import threading

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log
)  # for exponential backoff


class PromptExperiment:
    """
    This class is used to run a prompt experiment.
    """

    def __init__(self, api_key, prompts, aut_items, n_uses, example_df, n_trials, llm_params, n_examples=False,
                 by_quartile=False, title="", random_seed=416):
        self.api_key = api_key
        self.prompts = prompts
        self.aut_items = aut_items
        self.n_uses = n_uses
        self.example_df = example_df
        self.n_trials = n_trials
        self.n_examples = n_examples
        self.by_quartile = by_quartile
        self.random_seed = random_seed
        self.llm_params = llm_params  # temperature, frequency_penalty, presence_penalty
        self.title = title
        random.seed(self.random_seed)

        if self.n_examples:
            self.example_df['creativity_quartile'] = pd.qcut(self.example_df['target'], 4, labels=False) + 1

    def handle_prompt(self, args):
        """This function is used to run the experiment in parallel."""
        prompt_base, object_name, examples, temperature, frequency_penalty, presence_penalty, n_uses = args
        thread_id = threading.get_ident()
        logging.info(f"Thread {thread_id} started")
        prompt = self.make_prompt(prompt_base, object_name, examples, n_uses)
        response = self.generate_responses(prompt, temperature, frequency_penalty, presence_penalty)
        logging.info(f"Thread {thread_id} finished")

        return response, object_name

    def make_prompt(self, prompt_base, object_name, examples, n_uses):
        """This function replaces the placeholder text in the prompt template with the appropriate values.
        Params
        ------
        prompt_base: str - the prompt template
        object_name: str - the name of the object
        examples: list - a list of examples
        n_uses: int - the number of times the object will be used in the prompt

        Returns
            prompt: str - the prompt with the placeholder text replaced
        """
        prompt = prompt_base.replace("[OBJECT_NAME]", self.prepend_article(object_name))
        prompt = prompt.replace("[N]", str(n_uses))
        examples = " ".join(['\n- ' + item for item in examples]) + "\n"
        prompt = prompt.replace("[EXAMPLES]", examples)
        return prompt

    def get_examples(self, df, aut_item, quartile=None, seed=416):
        """This function returns a list of examples for a given aut_item, optionally filtered by quartile."""
        if quartile is not None:
            df = df.query(f'creativity_quartile == {quartile}')
        return df[df['prompt'] == aut_item].sample(self.n_examples, random_state=seed)['response'].tolist()

    @retry(wait=wait_random_exponential(multiplier=30, min=1, max=60), stop=stop_after_attempt(30),
           before_sleep=before_sleep_log(logging, logging.INFO))
    def generate_responses(self, prompt, temperature, frequency_penalty, presence_penalty):
        logging.info(f"Sending prompt to API: {prompt}")
        """This function generates a response from the OpenAI API.
        Params
        ------
        prompt: str
            The prompt to be sent to the API.
        temperature: float
            The temperature parameter for the API.
        frequency_penalty: float
            The frequency_penalty parameter for the API.
        presence_penalty: float
            The presence_penalty parameter for the API.
        Returns
        -------
        str
            The response from the API.
        """
        openai.api_key = self.api_key
        messages = openai.ChatCompletion.create(
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        msg = messages['choices'][0]['message']['content']
        # print(msg)
        return msg

    @staticmethod
    def prepend_article(object_name):
        """Prepend 'a ' or 'an ' to object_name based on its first letter and whether it's singular or plural."""
        vowels = 'aeiou'

        if object_name[-1] == 's':
            return object_name
        elif object_name[0].lower() in vowels:
            return 'an ' + object_name
        else:
            return 'a ' + object_name

    def run(self):
        # Setup logging and result file paths
        date_string = datetime.now().strftime("%Y-%m-%d__%H.%M.%S")
        log_file = f"{self.title or 'experiment'}_n{self.n_trials}_{date_string}.log"
        results_file = f"../../data/prompt_experiments/{self.title or 'results'}_n{self.n_trials}_{date_string}.jsonl"

        # Configure logging
        logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w',
                            format='%(asctime)s %(message)s')
        logging.info(
            f"Running {self.n_trials} trials with parameters: {self.llm_params}\nPrompts: {self.prompts}\nAUT ITEMS: {self.aut_items}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count() - 1, 12)) as executor, jsonlines.open(
                results_file,
                mode='w') as outfile:
            for prompt_name, prompt_base in self.prompts.items():
                prompt_args = []
                for trial in range(self.n_trials):
                    seed = self.random_seed + trial
                    random.seed(seed)
                    aut_item = random.choice(self.aut_items)
                    params = {key: random.choice(value) for key, value in self.llm_params.items()}
                    quartile = random.choice([1, 2, 3, 4]) if self.by_quartile else None
                    examples = self.get_examples(self.example_df, aut_item, quartile,
                                                 seed=trial) if '[EXAMPLES]' in prompt_base else []
                    prompt_args.append((prompt_base, aut_item, examples,
                                        *params.values(), self.n_uses, params))
                    logging.info("Args for trial {}: {}".format(trial, prompt_args[-1]))
                # Generate and record responses in parallel
                # We add params at the end to prompt_args just for data writing but it's redundant with temp, prescence, and freq
                # So it's hacky but that's why I remove the last element of `args'
                futures = [executor.submit(self.handle_prompt, args[:-1]) for args in prompt_args]
                for trial, future in enumerate(concurrent.futures.as_completed(futures)):
                    response, aut_item = future.result()  # get the aut_item from the future result
                    args = prompt_args[trial]
                    result = {**args[-1], 'aut_item': aut_item, 'prompt_condition': prompt_name, 'trial_no': trial,
                              'examples': args[2], 'output_responses': response, 'n_examples': len(args[2]),
                              'creativity_quartile': quartile if quartile is not None else 'N/A'}
                    outfile.write(result)
                logging.info(f"Processed all trials for prompt {prompt_name}")

        logging.info("Experiment completed")
