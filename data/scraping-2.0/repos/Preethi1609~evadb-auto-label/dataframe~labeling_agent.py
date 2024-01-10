import pandas as pd
import openai
# from config import Config
import json
import os
import tiktoken


class LabelingAgent(pd.DataFrame):
    def __init__(self, df, description=None, name=None) -> None:
        super().__init__(df)
        #initialize pandas dataframe
        self.df = df        
                
    def get_api_key(self):
        """ Initializes openai api with the openai key and model """
        open_ai_key = os.environ.get('OPENAI_API_KEY')
        openai.api_key = open_ai_key
        return

    def parse_config(self, config):

        # Load the JSON configuration from a file
        with open(config, 'r') as json_file:
            config_data = json.load(json_file)

        # Extract parameters for generating the LLM prompt
        self.task_type = config_data["task_type"]
        self.labels = ", ".join(config_data["prompt"]["labels"])
        self.task_guidelines = config_data["prompt"]["task_guidelines"].replace("{labels}", self.labels)
        self.output_guidelines = config_data["prompt"]["output_guidelines"].replace("{labels}", self.labels)
        self.few_shot_examples = config_data["prompt"]["few_shot_examples"]
        self.example_template = config_data["prompt"]["example_template"]
        self.label_column = config_data["dataset"]["label_column"]
        self.examples = "Some examples with their output answers are provided below:\n"
        
        seed_df = pd.read_csv(self.few_shot_examples)
        for _, row in seed_df.iterrows():
            example_values = [f"{val}" for col, val in row.items() if col != self.label_column]
            example = ', '.join(example_values)
            self.examples += self.example_template.replace("{example}", example).replace("{labels}", str(row[self.label_column]))
    
    def generate_prompt_classsification_task(self, show_question=False):        
        # Generate the LLM prompt
        question = "Now I want you to label the following comments:\n"
        for index, row in self.df.iterrows():
            current_example = "Input: "+ str(row['content']) + "\n"
            question += "{current_example}".replace("{current_example}", current_example)
        last_part = "Return the output in the same order as the comments"
        llm_prompt = f"{self.task_guidelines}\n{self.output_guidelines}\n{self.examples}\n{question}\n{last_part}"
        # print("LLM PROMT: ", llm_prompt)
        if show_question:
            print(question)
        return llm_prompt
    
    def check_price(self, config):
        self.c = self.parse_config(config) 
        prompt = self.generate_prompt_classsification_task()
        encoding = tiktoken.encoding_for_model(model_name="gpt-3.5-turbo")
        num_prompt_toks = len(encoding.encode(prompt))

        # max tokens
        num_label_toks = 100
        cost_per_prompt_token = 0.0015 / 1000
        cost_per_completion_token = 0.002 / 1000
        total_cost = (num_prompt_toks * cost_per_prompt_token) + (num_label_toks * cost_per_completion_token)
        print("Total cost of labeling would be: $", total_cost)
        return total_cost

    def label_data(self, config):
        self.c = self.parse_config(config) #create_labelling_prompt(config)
        prompt = self.generate_prompt_classsification_task(show_question=True)
        self.get_api_key()
        answer = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                                  temperature=0.2, \
                                                  messages=[{"role": "user", "content": prompt}]).choices[0].message.content
        return answer




    
    
    


