'''importing program modules'''
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import llm
# from .templates import prompt_temps
from langchain.prompts import PromptTemplate
load_dotenv()


class LangChainAI:
    '''Company Data analysis'''

    def __init__(self):
        self.open_ai_key = os.getenv("OPENAI_API_KEY")
        self.llm_ai = OpenAI(
            temperature=0.9,
            openai_api_key=self.open_ai_key
            )
        
    def content_generator(
            self,
            variable: str,
            template: str,
            ):
        ''' an content generator language model'''
        # Define the dynamic input variables
        input_variables = [var.strip() for var in variable.split(",")]
        for variable in input_variables:
            template += f" [Input- {variable}]: Describe {variable} in detail.\n"

        input_data = {}
        if len(input_variables) > 0:
            input_data = {i: input_variables[i] for i in range(len(input_variables))}

        # Create a Langchain prompt template
        prompt_template = PromptTemplate(
            input_variables=[],
            template=template
        )
        chained_llm = llm.LLMChain(
            llm=self.llm_ai,
            prompt=prompt_template
        )
        data = chained_llm.run(input_data)
        return data