from langchain import HuggingFaceHub
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
import warnings

warnings.filterwarnings("ignore")

class TextGen:
    def __init__(self):
        self.repo_id = "google/flan-t5-xxl"
        self.llm = HuggingFaceHub(repo_id=self.repo_id,
                             model_kwargs={"temperature": 0.1,
                                           "min_length": 150})
        
    def text_generator(self, input):
        # Chain 1: Origin
        prompt_template_name = PromptTemplate(
            input_variables=['input'],
            template="When was the term ({input}) found? Answer in a sentance.")
        origin_chain = LLMChain(llm=self.llm,
                                prompt=prompt_template_name,
                                output_key="Origin")

        # Chain 2: Description
        prompt_template_name = PromptTemplate(
            input_variables=['input'],
            template="State the definition of {input}.")
        desc_chain = LLMChain(llm=self.llm,
                              prompt=prompt_template_name,
                              output_key="Description")

        # Chain 3: Applications
        prompt_template_items = PromptTemplate(
            input_variables=['input'],
            template="What is the main applications of {input}?")
        application_chain = LLMChain(llm=self.llm,
                                     prompt=prompt_template_items,
                                     output_key="Applications")

        chain = SequentialChain(
            chains=[origin_chain, desc_chain, application_chain],
            input_variables=['input'],
            output_variables=['Origin', 'Description', "Applications"]
            )

        response = chain({'input': input})
        
        translate = PromptTemplate(
            input_variables=['input'],
            template="Translate ({input}) to French.")
        
        translated_response = {}
        for key in response:
            translate_chain = LLMChain(llm=self.llm, prompt=translate, output_key=key)
            translated_response[key] = translate_chain({'input': response[key]})[key]
        
        
        
        return response, translated_response