from langchain.chains import SequentialChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os


class ChainUtility:

    def __init__(self, input_variables, temperature=0):
        self.input_variables = input_variables
        self.temperature = temperature

    def get_chain(self, template, output_key):
        llm = ChatOpenAI(temperature=self.temperature, openai_api_key=os.getenv("OPENAI_KEY"))
        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt,
                         output_key=output_key
                         )
        return chain

    def get_overall_chain(self, chains, output_variables):
        overall_chain = SequentialChain(
            chains=chains,
            input_variables=self.input_variables,
            output_variables=output_variables,
            verbose=True
        )
        return overall_chain

    @staticmethod
    def print_completion(chain, start):
        resp = chain(start)
        print(resp)