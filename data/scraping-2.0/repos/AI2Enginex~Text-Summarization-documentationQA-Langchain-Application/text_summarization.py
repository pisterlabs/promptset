import os
import langchain
import openai
import warnings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# fro storing word vectors
from langchain.vectorstores import FAISS
# for getting the documents
from langchain.chains import RetrievalQA
# for loading the document
from langchain.document_loaders import TextLoader
# converting our text to vectors
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
import pandas as pd
import json
warnings.filterwarnings('ignore')
os.environ['OPENAI_API_KEY'] = 'xx-xxxx'
key = os.environ.get('OPENAI_API_KEY')


# creating Open AI object class
class OpenAIObject:

    def __init__(self, model=None):

        if model is not None:
            self.llm = OpenAI(temperature=0, openai_api_key=key, model=model)
        else:
            self.llm = OpenAI(temperature=0, openai_api_key=key)


class ChatModelAPI:

    def __init__(self):
        self.chat_model = ChatOpenAI(
            temperature=0, openai_api_key=key, max_tokens=1000)


# creating summarisation class for
# creating summary for any given text
class TextSummarisation(OpenAIObject):

    def __init__(self, model=None):
        super().__init__(model)

    # function for cretaing instruction template
    # this function is used to pass instruction
    # to GPT model for displaying the output

    def command_instructions(self):

        try:
            template = '''
             %INSTRUCTIONS:
             please summarize the following piece of text.
             respond in a manner that a 5 years old can understand.
             
             %TEXT:
             {text}
             '''
            return PromptTemplate(input_variables=["text"],    template=template)

        except Exception as e:
            pass

    # function to create summary of any given context
    # generates the answers as per the instructions given before
    def summarise_text(self, user_input):

        try:
            prompt = self.command_instructions()
            final_prompt = prompt.format(text=user_input)
            return self.llm(final_prompt)
        except Exception as e:
            return e

# class for generating summary for long text


class LongTextSummarisation(OpenAIObject):

    def __init__(self, model=None):

        super().__init__(model=model)

    # creating documents for the large text inputs
    # passing the input as batch to the model
    def generate_docs(self, text):

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=[".", ","], chunk_size=5000, chunk_overlap=300)
            return text_splitter.create_documents([text])
        except Exception as e:
            return e

    # summarizing each batch input
    def summarise_long_text(self, user_input):

        try:
            chain = load_summarize_chain(
                llm=self.llm, chain_type='map_reduce', verbose=True)
            return chain.run(self.generate_docs(user_input))
        except Exception as e:
            return e


# class for Generating Summary for an
# entire document
class DocumentSummarisation(OpenAIObject):

    def __init__(self, model=None, filepath=None):

        super().__init__(model=model)
        self.file = filepath

    # loading the text file
    def load_file(self):

        try:
            text = open(self.file, 'r', encoding='unicode_escape')
            return text.read()
        except Exception as e:
            return e

    # generating docs for each line
    # of the file
    def generating_docs(self):

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"], chunk_size=3000, chunk_overlap=300)
            return text_splitter.create_documents([self.load_file()])
        except Exception as e:
            return e

    # summarizing the docs and generating
    # overall summary
    def summarise_doc_text(self):

        try:
            chain = load_summarize_chain(llm=self.llm, chain_type='map_reduce')
            return chain.run(self.generating_docs())
        except Exception as e:
            return e


# class for Questning Answering
# fro the given document
class DocumentQA(OpenAIObject):

    def __init__(self, model=None, filepath=None):
        super().__init__(model)
        self.file = filepath

    # function for loading file
    def load_file(self):

        try:
            loader = TextLoader(self.file, encoding='unicode_escape')
            return loader.load()
        except Exception as e:
            return e

    # function for creating batch inputs
    def create_batch(self):

        try:
            textsplitter = RecursiveCharacterTextSplitter(
                chunk_size=2500, chunk_overlap=250)
            return textsplitter.split_documents(self.load_file())
        except Exception as e:
            return e

    # function for generating embaddings
    def get_embeddings(self):
        try:
            return OpenAIEmbeddings(openai_api_key=key)
        except Exception as e:
            return e

    def command_qa_instructions(self):
        try:
            template = '''
              %INSTRUCTIONS:
              Answer the following question only based on the document content, also if any persons name or service name occurs,
              first search the name in the document and then provide the answer with respect to document content
          
              %TEXT:
              {text}
              '''
            return PromptTemplate(input_variables=["text"],    template=template)
        except Exception as e:
            pass

    def prompt_template_design(self, query):
        try:
            prompt = self.command_qa_instructions()
            return prompt.format(text=query)
        except Exception as e:
            return e

    def run_engine(self, query):
        try:
            doc_batches = self.create_batch()
            docsearch = FAISS.from_documents(
                doc_batches, self.get_embeddings())
            qa = RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=docsearch.as_retriever())
            return (qa.run(self.prompt_template_design(query)))
        except Exception as e:
            return e


# class for predicting context for
# a given line in the file
class ContextMapping(ChatModelAPI):

    def __init__(self, filepath='E:\MoneyControl Data Scraping\LinkedInIndustries.csv'):
        super().__init__()
        self.df = pd.read_csv(filepath)

    # generating mappings structure
    # for input and output

    def get_mapping_structure(self):

        try:
            response_schemas = [
                ResponseSchema(
                    name="input_industry", description="This is the input_industry from the user"),
                ResponseSchema(name="standardized_industry",
                               description="This is the industry you feel is most closely matched to the users input"),
                ResponseSchema(
                    name="match_score",  description="A score 0-100 of how close you think the match is between user input and your match")
            ]
            return StructuredOutputParser.from_response_schemas(response_schemas)
        except Exception as e:
            return e

    # parsing the structure
    def parsing_output_structure(self):
        try:
            return self.get_mapping_structure().get_format_instructions()
        except Exception as e:
            return e

    # this tells chat gpt what input
    # is given and how to generate the output
    def generating_prompt_structure(self):

        try:
            template = """
                      You will be given a series of industry names from a user.
                      Find the best corresponding match on the list of standardized names.
                      The closest match will be the one with the closest semantic meaning. Not just string similarity.
                      
                      {format_instructions}
                      
                      Wrap your final output with closed and open brackets (a list of json objects)
                      
                      input_industry INPUT:
                      {user_industries}
                      
                      STANDARDIZED INDUSTRIES:
                      {standardized_industries}
                      
                      YOUR RESPONSE:
                      """

            return ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template(template)
                ],
                input_variables=["user_industries", "standardized_industries"],
                partial_variables={
                    "format_instructions": self.parsing_output_structure()}
            )
        except Exception as e:
            return e

    # generating the context
    def get_context(self, userinput):

        try:
            _input = self.generating_prompt_structure().format_prompt(user_industries=userinput,
                                                                      standardized_industries=", ".join(self.df['Industry'].values))
            return self.chat_model(_input.to_messages()).content

        except Exception as e:
            return e


if __name__ == '__main__':
    pass
    # t = TextSummarisation(model='text-davinci-003')
    # print(t.summarise_text(user_input="The market continued its uptrend journey for yet another session with the Nifty50 scaling past the 19,800 mark, taking support at 19,750 throughout the day on October 11. Hence, as long as the index holds the 19,800-19,750 area, reaching 20,000 points seems possible in the coming sessions, whereas on other side, the 19,700-19,600 zone is likely to act as a support in case of correction, experts said.The Nifty50 jumped 122 points to 19,811, and the BSE Sensex climbed 394 points to 66,473, while market breadth remained positive in the ratio of 2:1. The Nifty Midcap 100 and Smallcap 100 indices also participated in the run, rising half a percent and eight-tenth of a percent"))

    # l = LongTextSummarisation(model=None)
    # print(l.summarise_long_text(user_input="Prabhudas Lilladher has come out with its second quarter (July-September’ 24) earnings estimates for the Consumer Durables sector. The brokerage house expects Bajaj Electricals to report net profit at Rs. 58.5 crore down 8.5% year-on-year (up 35.9% quarter-on-quarter).Net Sales are expected to decrease by 0.4 percent Y-o-Y (up 9.3 percent Q-o-Q) to Rs. 1,215.7 crore, according to Prabhudas Lilladher.Earnings before interest, tax, depreciation and amortisation (EBITDA) are likely to fall by 5.4 percent Y-o-Y (up 28 percent Q-o-Q) to Rs. 88.7 crore.Disclaimer: The views and investment tips expressed by investment experts on moneycontrol.com are their own, and not that of the website or its management. Moneycontrol.com advises users to check with certified experts before taking any investment decisions."))

    # ds = DocumentQA(filepath='E:\MoneyControl Data Scraping\data.txt')
    # print(ds.run_engine("what is the document all about"))

    # doc_summary = DocumentSummarisation(filepath='E:\MoneyControl Data Scraping\data.txt')
    # print(doc_summary.summarise_doc_text())
