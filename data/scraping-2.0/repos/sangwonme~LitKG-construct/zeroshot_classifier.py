# input: categories and its definition
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import ast
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class ZeroshotClassifier:
    def __init__(self, categories, abstracts):
        # Load the SpaCy model
        self.nlp = spacy.load("en_core_web_sm")
        # Set up the turbo LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model_name='gpt-3.5-turbo',
        )
        print('LLM setup is done!')
        # abstract data with splitted sentences
        self.abstracts = []
        self.split_sentences_abstract(abstracts)
        print('Splitting abstract sentences is done!')
        # categories output
        self.categories = categories

    # Function to extract noun phrases with their locations
    def split_sentences_abstract(self, abstracts):
        for paper_index, abstract in enumerate(abstracts):
            doc = self.nlp(abstract)
            sentences = []
            for sent_index, sent in enumerate(doc.sents):
                sentences.append('\"'+str(sent)+'\"')
            self.abstracts.append(sentences)
            print('Abstract', paper_index, 'is splitted.')
    
    # Classifier
    def classification(self):
        classification_result = []
        categories_description = ''
        for category in self.categories.keys():
            categories_description += f'- {category}: {self.categories[category]} \n'
        for paper_index, abstract in enumerate(self.abstracts):
            messages = [
                SystemMessage(
                    content="You are a helpful assistant that classify the sentences in given categories."
                ),
                HumanMessage(content=f"""
                I will give you abstract of research paper.
                Your role is to classify each sentence in one of following categories.
                {categories_description}

                Could you label all sentences in the abstract one by one please?
                - Do not give the description just following answer
                - Make sure that input's length and output's length should be same!

                <Input Format>
                ['sentence 1', 'sentence 2', ... , 'sentence N'] (length of the input = N)
                <Output Format>
                ['category for sentence 1', 'category for sentence 2', '....']

                <Abstract>
                {str(abstract)} (length = {len(abstract)})
                """),
            ]
            print(abstract)
            print('len :', len(abstract))
            result = self.llm(messages)
            result = ast.literal_eval(result.content)
            classification_result.append(result)
            print('Paper', paper_index, 'classified.')
            
        return classification_result
