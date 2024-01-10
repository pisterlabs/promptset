import openai
import random
import pandas as pd
import numpy as np
import os
import json
from keys import OPENAI_API_KEY
from construction_format import ConstructionFormat, ArrowFormat, QAFormat

class APIAccess:
    """
    Access the OpenAI API and obtain probabilities for each token in a generated prompt.
        
    Attributes:
        prompt (Prompt): the prompt for which to calculate token probabilities
        parspeed_prompt_df (DataFrame): the prompt as a DataFrame 
    """
    def __init__(self, prompt):
        self.prompt = prompt
        self.parsed_prompt_df = pd.DataFrame([e.as_dict() for e in self.prompt.examples])
    
    def request(self, model, format, needs_instruction):
        """
        Query the API with the generated prompt and retrieve an output of the probabilities of each token
            
        Args:
            model (str): the OpenAI model to query with generated prompt
            format (str): the format of the prompt ['arrow', 'qa']
            needs_instruction (bool): True if need to include instruction in prompt and False otherwise
        Returns:
            output (openai.openai_object.OpenAIObject): output from OpenAI API
        """
        prompt = self.generate_formatted_prompt(format, needs_instruction, to_togethercomputer=False)
        openai.api_key = OPENAI_API_KEY
        output = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=0, 
            logprobs=4,
            echo=True,
        )
        return output

    def generate_data_for_openai_finetuning(self, format, needs_instruction): 
        """
        Skips quering the API and instead creates a file containing information necessary for finetuning the model
        
        Args:
            format (str): the desired format ['arrow', 'qa']
            needs_instruction (bool): True if need to include instruction in prompt and False otherwise
        Returns:
            None
        """
        prompt_and_completion = self.generate_formatted_prompt(format, needs_instruction, to_togethercomputer=False)
        
        if format == 'arrow':
            prompt = prompt_and_completion[:-1]
            completion = prompt_and_completion[-1:]
        else:
            prompt = prompt_and_completion[:-2]
            completion = prompt_and_completion[-2:]
            
        formatted_generation = {
            "prompt": prompt,
            "completion": completion
        }

        filepath = f"for_finetuning/test.jsonl"

        with open(filepath, 'a+') as f:
                f.write(json.dumps(formatted_generation))
                f.write('\n')
        
    def to_togethercomputer(self, format, request_type, model, needs_instruction, max_tokens, logprobs):
        '''
        Skips quering the API and instead creates a file containing information necessary for querying TogetherComputer (t0pp) via Stanford internal API

        Args:
            format (str): the desired format ['arrow', 'qa']
            request_type (str): the type of request to send to t0pp, here always "language-model-inference"
            model (str): the model to query with generated prompt (t0pp)
            needs_instruction (bool): True if need to include instruction in prompt and False otherwise
            max_tokens (int): the number of tokens to generate
            logprobs (int): the number of logprobs to return

        Returns:
            None
        '''
        prompt_list, solutions = self.generate_formatted_prompt(format, needs_instruction, to_togethercomputer=True)
        for prompt in prompt_list:
            request = {
                "request_type": request_type, 
                "model": model, 
                "prompt": prompt, 
                "max_tokens": max_tokens, 
                "logprobs": logprobs
                }
            filepath = f"togethercomputer/for_rebuttal.jsonl"

            with open(filepath, 'a+') as f:
                f.write(json.dumps(request))
                f.write('\n')
            
        solutions.to_csv(f"togethercomputer/for_rebuttal_solutions.csv", mode='a', header=False)
        
    def format_constructions(self, format):
        """
        Append correct format to prompt prior to querying the API. 
        Includes prefixes (before the prompt), infixes (between the prompt and the answer), and suffixes (after the answer).

        For example, if format == 'qa' : Sentence | Label --> Q: Sentence\nA: Label
            
        Args:
            format (str): the desired format ['arrow', 'qa']
        Returns:
            None
        """
        construction_format = self.get_format_class(format)
        affixes = construction_format.get_affixes()
        self.parsed_prompt_df['XY_relabeled'] = np.where(self.parsed_prompt_df.active_task_label == True, 'X', 'Y')
        self.parsed_prompt_df['formatted_construction'] = affixes[0] + self.parsed_prompt_df['construction'] + affixes[1] + self.parsed_prompt_df['XY_relabeled'] + affixes[2]

    def get_format_class(self, format_name):
        """
        Select the desired format for the prompt ['qa', 'arrow']

        'qa': Sentence, Label --> "Q: Sentence\nA: Label"
        'arrow': Sentence, Label --> "Sentence\n> Label"

        Args:
            format_name (string): the name of the desired format
        Returns:
            Subclass corresponding to the desired format (e.g. ArrowFormat, QAFormat)
        """
        formats = {
            'arrow' : ArrowFormat(),
            'qa' : QAFormat()}
        
        if format_name in formats:
            construction_format = formats[format_name]
            return construction_format
        
        raise Exception("invalid format type")

    def generate_formatted_prompt(self, format, needs_instruction, to_togethercomputer):
        """
        Formats constructions for the API query and adds new line between consecutive examples.
        
        For example, if format == 'arrow':
        Construction 1
        > Label 1
        Construction 2
        > Label 2
        Query
        > Query Label

        Args:
            format (str): the desired format
            needs_instruction: True if need to includ instruction in prompt and False otherwise
            to_togethercomputer: True if need to format prompt for togethercomputer (e.g. for t0pp) and False otherwise
        Returns:
            (str): formatted prompt as a single string for API query
        """
        self.format_constructions(format)
        if not to_togethercomputer:
            if needs_instruction:
                return self.prompt.get_instruction() + '\n' + self.parsed_prompt_df['formatted_construction'].str.cat(sep='\n')
            return self.parsed_prompt_df['formatted_construction'].str.cat(sep='\n')
        else:
            construction_list = self.parsed_prompt_df['formatted_construction'].to_list()
            
            if needs_instruction:
                construction_list[0] = self.prompt.get_instruction() + '\n' + construction_list[0]
            for i in range(1, len(construction_list)):
                construction_list[i] = construction_list[i-1] + '\n' + construction_list[i]
            sols = pd.DataFrame()
            if format == 'qa':   
                sols['solution'] = [construction[-2:] for construction in construction_list]
                construction_list = [construction[:-2] for construction in construction_list]
            elif format == 'arrow':
                sols['solution'] = [construction[-1:] for construction in construction_list]
                construction_list = [construction[:-1] for construction in construction_list]
            else:
                raise ValueError('invalid format')
            return (construction_list, sols)
    
    def to_numpy_dataframe(self, output):
        """
        Reformat the output of the API into a numpy dataframe
        
        Args:
            output (openai.openai_object.OpenAIObject): output from API query
        Returns:
            unpacked_df (pd.DataFrame): the DataFrame obtained from the API call
        """
        # modified from http://gptprompts.wikidot.com/intro:logprobs
        
        unpacked_df = pd.DataFrame(output["choices"][0]["logprobs"])

        unpacked_df = unpacked_df.drop(columns=['text_offset'])
        unpacked_df["%"] = unpacked_df["token_logprobs"].apply(lambda x: 100*np.exp(x))
        
        return unpacked_df 

    def isolate_probs(self, df):
        """
        Removes all columns that are not the tokens and the corresponding probabilities

        Args:
            df (pd.DataFrame): dataframe with cleaned data from API query
        Returns:
            df (pd.DataFrame): a dataframe will all superfluous columns removed
        """
        df.drop(df.columns.difference(['tokens', '%', 'top_logprobs', 'index']), 1, inplace = True)
        return df

    def save_to_file(self, model, construction_format, construction_type, shots, unpacked_df, iteration=0):
        file_name = f"{model}/{construction_type}/{construction_format}_{shots}_{iteration}.csv"
        unpacked_df.to_csv(file_name, mode='a', header=not os.path.exists(file_name))

    

