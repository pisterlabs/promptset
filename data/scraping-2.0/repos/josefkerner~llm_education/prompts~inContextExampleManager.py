from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector, SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from typing import List,Dict
import os, re, pandas
class inContextExampleManager:
    def __init__(self):
        if os.name == "nt":
            datapath= "data/train.csv"
        else:
            datapath = "/app/train.csv"
        self.examples = self.load_from_file(datapath)

    def load_from_file(self, file_path) -> List[Dict]:
        '''
        Loads examples from CSV file
        :param file_path:
        :return:
        '''
        df = pandas.read_csv(file_path, sep=';', encoding='utf-8-sig')
        return df.to_dict('records')
    def get_examples(self, type: str):
        '''
        It needs to only have attributes contained in the specified prompt
        :param type:
        :return:
        '''
        message_examples = [
            {
                'name': example['name'],
                'message': example['message'],
                'cv_summary': example['cv_summary'],
                'role_description': example['role_description']
            }
            for example in self.examples
        ]

        rec_examples = [
            {
                'name': example['name'],
                'surname': example['surname'],
                'cv_summary': example['cv_summary'],
                'role_description': example['role_description'],
                'rec': example['rec'],
            }
            for example in self.examples
        ]
        question_examples = [
            {
                # 'message': example['message'],
                'cv_summary': example['cv_summary'],
                'role_description': example['role_description'],
                #'rec': example['rec'],
                'question': example['question']
            }
            for example in self.examples
        ]
        if type == 'message':
            return message_examples
        elif type == 'rec':
            return rec_examples
        elif type == 'question':
            return question_examples
        else:
            raise ValueError(f"{type} is not a valid example type")

    def get_prompt(self, name: str, surname: str, cv_summary: str, role_summary: str):
        '''
        Generates a prompt for the message generation task
        :param name:
        :param surname:
        :param cv_summary:
        :param role_summary:
        :return:
        '''
        prompt = self.select_examples(
            input_variables=["name", "surname", "role_description", "cv_summary", "rec"],
            selection_strategy="max_marginal",
            prefix=""""
            Generate a recommendation if candidate a fit for the job role based on role requirements and candidate experience.
            If a candidate is not a fit for the job role, generate only string : "Not a fit".
            """,
            suffix="""
            Candidate full name: {name}{surname}\n Candidate experience: {cv_summary}\n Job role description: {role_description}\n Generated recommendation: {rec}
            """,
            type='rec',
            num_examples=5

        )

        prompt = prompt.format(
            name=name,
            surname=surname,
            cv_summary=cv_summary,
            role_description=role_summary,
            rec=""
        )
        return prompt

    def get_max_marginal_selector(self, examples : List[Dict], k: int):
        '''
        Returns a MaxMarginalRelevanceExampleSelector
        :param examples:
        :param k:
        :return:
        '''
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            # This is the list of examples available to select from.
            examples,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            OpenAIEmbeddings(),
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=k
        )
        return example_selector
    def select_examples(self,selection_strategy:str,
                        input_variables: List[str],
                        prefix: str, suffix: str,
                        type: str,
                        num_examples: int = 2) -> FewShotPromptTemplate:
        '''
        Returns a FewShotPromptTemplate
        :param selection_strategy:
        :param input_variables:
        :param prefix:
        :param suffix:
        :param type:
        :param num_examples:
        :return:
        '''

        if os.name == 'nt':
            os.environ['REQUESTS_CA_BUNDLE'] = "C:/python/openai/openai.crt"

        examples = self.get_examples(type=type)
        example_prompt = PromptTemplate(
            input_variables=input_variables,
            template=suffix,
        )

        if selection_strategy == "max_marginal":
            example_selector = self.get_max_marginal_selector(examples=examples,
                                                              k=num_examples)
        else:
            raise ValueError(f"{selection_strategy} example selection strategy is not implemented")

        similar_prompt = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
        )
        return similar_prompt


