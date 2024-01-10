from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

import os
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class Extract():
    def __init__(self, para, entities):
        self.para = para
        self.entities = entities


# chain to extract entities from paragraph
    def extract_entities(self):
        llm = OpenAI(temperature=0.0)
        extract_template = """
        Given the following paragraph: {paragraph}, extract the relevant {entities}
        """
        prompt_template = PromptTemplate(
            input_variables=["paragraph", "entities"],
            template=extract_template
        )
        extract_chain = LLMChain(
            llm=llm, prompt=prompt_template, output_key="list", verbose=False
        )

        output = extract_chain({"paragraph": self.para, "entities": self.entities})
        return output.get('list')
    
# chain to remove parentheses from entities list
    def remove_paras(self):
        sentence = self.para
        return_sentence = sentence
        new_sentence = ''
        if '(' in sentence:
            loc = sentence.find('(')
            for i in range(loc, len(sentence)):
                if sentence[i] == ')':
                    new_sentence += sentence[i]
                    break
                else:
                    new_sentence += sentence[i]
                    continue
            return_sentence = sentence.replace(new_sentence, '')
            return return_sentence
        else:
            return return_sentence


