import yaml
import os
from tqdm import tqdm
import re

import pandas as pd

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate


class CodeConditionedGenerator:
    def __init__(self, credentials_path, data_path):
        with open(credentials_path, "r") as f:
            credentials = yaml.load(f, Loader=yaml.FullLoader)

        os.environ["AZURE_OPENAI_ENDPOINT"] = credentials['AZURE_OPENAI_ENDPOINT']
        os.environ["AZURE_OPENAI_API_KEY"] = credentials['AZURE_OPENAI_API_KEY']

        self.model = AzureChatOpenAI(
            openai_api_version="2023-05-15",
            azure_deployment=credentials['Deployment_completion'],
        )

        self.df = pd.read_csv(data_path)

    def generate_docstrings(self):
        prompt = '''
Given the code of the {type} below your taks is to generate docString describing functions inside.
Firstly pay attention to all variables that
are used in the code. Secondly, analyze what is function doing with those variables.
Based on this, deduce what steps are being taken in this function and what purpose they serve.
Having all that informations gathered in your mind, 
write a docstring for the function that will describe step by step what is this function doing, what's it's purpose and what variables it is affecting.
Do not write anything other than the docstring, docstring should be the only output.
Start your answer with ```python\n"""\n and end it with """\n```
\n\n{code}\n\n
'''

        chat_prompt_template = ChatPromptTemplate.from_template(prompt)

        mask_docstringable = self.df['type'] != 'module'
        df_docstringable = self.df.loc[mask_docstringable]
        prompts = [chat_prompt_template.format_prompt(type=example['type'], code=example['code']) for _, example in df_docstringable.iterrows()]

        docstrings = []

        for prompt in tqdm(prompts):
            result = self.model(prompt.to_messages()).content
            result = re.sub('^\n*```\n*(python)\n*(""")?\n*', '', result)
            result = re.sub('\n*(""")?\n*```\n*$', '', result)
            docstrings.append(result)

        self.df.loc[mask_docstringable, 'generated_docstring'] = docstrings
        return self.df
    
    def generate_descriptions(self):
        raise NotImplementedError
    
    def save_df(self, path):
        self.df.to_csv(path, index=False)

        