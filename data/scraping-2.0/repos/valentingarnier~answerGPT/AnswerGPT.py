import streamlit as st
import time
from langchain.chains import SequentialChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from datetime import datetime
from langchain.llms import OpenAI
from langchain.output_parsers import DatetimeOutputParser
from langchain.chat_models import ChatOpenAI
from openai import OpenAIError

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate


class AnswerGPT:
    def __init__(self, api_key, synthetic_level, tone, original_message):
        self.api_key = api_key
        self.synthetic_level = synthetic_level
        self.tone = tone
        self.original_message = original_message
        self.chain = None
        self.email_summary = None

    def define_instructions(self):
        # Define the common instructions for the language model
        system_instructions = """
            You are a helpful AI assistant with expertise in crafting email responses.
            Your responses should be clear, utilizing proper structured techniques like bullet points, and paragraph breaks where needed.
            You will respond in the language of the email you must respond to.
        """

        # Define the specific instructions based on user input
        system_instructions += "\n{synthetic_level}"
        system_instructions += "\nMaintain a {tone} tone."

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_instructions)
        human_message_prompt = HumanMessagePromptTemplate.from_template("Reply to the following email:\n{email}")
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        return chat_prompt

    def generate_summary(self):
        try:
            # Initialize the language model
            chat = ChatOpenAI(openai_api_key=self.api_key, model_name='gpt-4')
            # Combine the common and specific instructions
            chat_prompt = self.define_instructions()
            # Prepare the prompt for the language model
            request = chat_prompt.format_prompt(email=self.original_message,
                                         synthetic_level=self.synthetic_level,
                                         tone=self.tone
                                         ).to_messages()
            response = chat(request)

            prompt_template = """Write a concise summary with bullet points of the following:
            "{email}".\n
            This email is directed to me so summarize this email for me. I am the receiver.
            CONCISE SUMMARY:"""
            prompt = PromptTemplate.from_template(prompt_template)

            # Define LLM chain
            llm = ChatOpenAI(temperature=0, model_name="gpt-4")
            chain_summarize = LLMChain(llm=llm, prompt=prompt, output_key="email_summary")

            chain_summary = SequentialChain(
                chains=[chain_summarize],
                input_variables=["email"],
                output_variables=["email_summary"],
                verbose=True,
            )

            results = chain_summary({"email": self.original_message})
            email_summary = results["email_summary"]

            template = """
            You are a helpful assistant that answers my emails.\n
            You need to craft the perfect answer for me.\n
            Here's the email summary you need to craft an answer for: {email_summary}\n

            Use this context representing additional information: 

            {chat_history}

            {context}

            {query}
            """

            prompt = PromptTemplate(
                input_variables=["email_summary", "chat_history", "context"],
                template=template,
            )

            memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")

            chain = load_qa_chain(
                ChatOpenAI(temperature=0, model_name="gpt-4"), chain_type="stuff", memory=memory, prompt=prompt
            )
            self.chain = chain
            self.email_summary = email_summary
            return email_summary

        except OpenAIError as e:
            return e
        except Exception as e:
            return e
    def ask_question(self):
        query = """
                        Ask the most relevant question to craft an answer. Ask the question in a concise manner.
                        If there is no need for answers from the receiver in the email, simply output NO.
                        Don't take any decision without asking a question first.
                    """
        res = self.chain(
            {
                "input_documents": [],
                "query": query,
                "email_summary": self.email_summary,
            },
            return_only_outputs=True,
        )
        return res['output_text']

    def craft_answer(self):
        chat_prompt = self.define_instructions()
        request = chat_prompt.format_prompt(email=self.original_message,
                                            synthetic_level=self.synthetic_level,
                                            tone=self.tone,
                                            ).to_messages()

        query = request[0].content
        email_answer = self.chain(
            {
                "input_documents": [],
                "query": query,
                "email_summary": self.email_summary,
            },
            return_only_outputs=True,
        )
        return email_answer['output_text']
