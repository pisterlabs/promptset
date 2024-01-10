# -*- coding: utf-8 -*-
"""
    LangChain Quickstart
    ~~~~~~~~~~~~~~~~~~~~
    LangChain Explained in 13 Minutes | QuickStart Tutorial for Beginners

    see: https://www.youtube.com/watch?v=aywZrzNaKjs
         https://github.com/rabbitmetrics/langchain-13-min
"""
import os

import pinecone
from dotenv import find_dotenv, load_dotenv

# 5.) sequential chains
# 4.) chains
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# 1.) wrappers
from langchain.llms.openai import OpenAI

# 3.) prompt templates
from langchain.prompts import PromptTemplate
from langchain.python import PythonREPL

# 2.) models and messages
from langchain.schema import HumanMessage, SystemMessage  # AIMessage (not used)

# 6.) embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 7.) pinecode client
from langchain.vectorstores.pinecone import Pinecone

# 8.) LangChain agents
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent

from models.conf import settings


# Load environment variables from .env file in all folders
# pylint: disable=duplicate-code
dotenv_path = find_dotenv()
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, verbose=True)
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_API_ORGANIZATION = os.environ["OPENAI_API_ORGANIZATION"]
else:
    raise FileNotFoundError("No .env file found in root directory of repository")


class LangChainDev:
    """LangChain Quickstart"""

    PINECONE_INDEX_NAME = "langchain-quickstart"

    multi_prompt_explanation = None
    texts_splitter_results = None
    pinecone_search = None
    openai_embedding = OpenAIEmbeddings(model_name="ada")  # minute: 10:05
    query_result = None
    agent_executor = create_python_agent(  # minute: 11:45
        llm=OpenAI(temperature=0, max_tokens=1000),
        tool=PythonREPL(),
        verbose=True,
    )
    # pylint: disable=no-member
    pinecone.init(
        api_key=settings.pinecone_api_key.get_secret_value(), environment=settings.pinecone_environment
    )  # minute 10:43

    # LLM wrappers. minute 5:46
    def test_01_basic(self):
        """Test a basic request"""

        llm = OpenAI(model_name="text-davinci-003")
        retval = llm("explain large language models in one sentence")
        print(retval)

    # 2.) models and messages. minute 6:08
    def test_02_chat_model(self):
        """Test a chat model"""
        chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
        messages = [
            SystemMessage(content="You are an expert data scientist"),
            HumanMessage(content="Write a Python script that trains a neural network on simulated data"),
        ]
        retval = chat(messages)
        print(retval.content, end="\n")

    # 3.) prompt templates. minute 6:56
    def get_prompt(self):
        """Get a prompt"""
        template = """
        You are an expert data scientist with an expertise in building deep learning models.
        Explain the concept of {concept} in a couple of lines.
        """
        prompt = PromptTemplate(input_variables=["concept"], template=template)
        return prompt

    def test_03_prompt_templates(self):
        """Test prompt templates"""
        llm = OpenAI(model_name="text-davinci-003")
        prompt = self.get_prompt()
        retval = llm(prompt.format(concept="regularization"))
        print(retval)

    # 4.) chains. minute 7:45
    def get_chain(self, llm, prompt):
        """Get a chain"""
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain

    def test_04_chain(self):
        """Test a chain"""
        llm = OpenAI(model_name="text-davinci-003")
        prompt = self.get_prompt()
        chain = self.get_chain(llm=llm, prompt=prompt)
        print(chain.run("autoencoder"))

    # 5.) sequential chains. minute 8:06
    def get_overall_chain(self, chains):
        """Get an overall chain"""
        return SimpleSequentialChain(chains=chains, verbose=True)

    def get_prompt_two(self):
        """Get a second prompt"""
        second_prompt = PromptTemplate(
            input_variables=["ml_concept"],
            template="""
            Turn the concept description of {ml_concept} and explain it to me like I'm five in 500 words.
            """,
        )
        return second_prompt

    def get_explanation(self):
        """Get an explanation"""
        llm = OpenAI(model_name="text-davinci-003")
        prompt = self.get_prompt()
        chain_one = self.get_chain(llm=llm, prompt=prompt)

        second_prompt = self.get_prompt_two()
        chain_two = self.get_chain(llm=llm, prompt=second_prompt)
        overall_chain = self.get_overall_chain(chains=[chain_one, chain_two])
        return overall_chain.run("autoencoder")

    def test_05_chains(self):
        """Test chains"""
        self.multi_prompt_explanation = self.get_explanation()
        print(self.multi_prompt_explanation)

    # 6.) embeddings. minute 9:00
    def test_06_embeddings(self):
        """Test embeddings"""
        # minute 9:32
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
        )
        self.multi_prompt_explanation = self.get_explanation()
        if not self.texts_splitter_results:
            self.texts_splitter_results = text_splitter.create_documents([self.multi_prompt_explanation])
            print(self.texts_splitter_results[0].page_content)

    # minute 10:05
    def test_06_embeddings_b(self):
        """Test embeddings b"""
        if not self.query_result:
            self.query_result = self.openai_embedding.embed_query(  # minute 10:21
                self.texts_splitter_results[0].page_content
            )
            print(self.query_result)

        # 7.) pinecone client. minute 11:00
        self.pinecone_search = Pinecone.from_documents(
            documents=self.texts_splitter_results,
            embedding=self.openai_embedding,
            index_name=self.PINECONE_INDEX_NAME,
        )

    # pinecone (continued). minute 11:12
    def test_07_pinecone_search(self):
        """Test pinecone search"""
        query = "What is magical about an autoencoder?"
        result = self.pinecone_search.similarity_search(query)
        print(result)

    # 8.) LangChain agents. minute 11:45
    #     (unrelated.)
    def test_08_agent_executor(self):
        """Test agent executor"""
        retval = self.agent_executor.run("Find the roots (zeros) of the quadratic function 3 * x**2 + 2*x -1")
        print(retval)

    def main(self):
        """Main function"""
        # self.test_06_embeddings()
        # self.test_06_embeddings_b()
        # self.test_07_pinecone_search()
        # self.test_08_agent_executor
        self.test_03_prompt_templates()


def main():
    """Main function"""
    pintcode_tests = LangChainDev()
    pintcode_tests.main()


if __name__ == "__main__":
    main()
