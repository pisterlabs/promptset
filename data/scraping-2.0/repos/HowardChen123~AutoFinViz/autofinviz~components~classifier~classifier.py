import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

class Classifier():

    def __init__(self, model="gpt-3.5-turbo") -> None:
        self.model = ChatOpenAI(model_name=model)
        
        with open('autofinviz/components/classifier/prompts/classifier.tmpl', 'r') as file:
            prompt_template_content = file.read()

        self.prompt_template = ChatPromptTemplate.from_messages([("system", prompt_template_content), ("human", "{input}")])
        self.output_parser = StrOutputParser()

    def validate_output(self, text):
        valid_tags = ['Market Dataset', 'Economic Dataset', 'Corporate Financial Dataset']
        for tag in valid_tags:
            if tag in text:
                return f"{tag}"
        raise ValueError("Output does not contain a valid dataset tag")

    def classify(self, df: pd.DataFrame):
        col = df.columns

        input = f"""
        Here are the column names of a dataset: {list(col)}. Based on these names, classify the dataset into one of the specified categories.
        """
        chain = self.prompt_template | self.model | self.output_parser
        return self.validate_output(chain.invoke({"input": input}))