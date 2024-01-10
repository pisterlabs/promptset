# ChatBOT - the ChaatBOT maain class.
#
# This class represents a generic chatbot. It is composed by:
# - a model
# - a chat history
#
# Copyright (C) 2023 Salvatore D'Angelo
# Maintainer: Salvatore D'Angelo sasadangelo@gmail.com
#
# SPDX-License-Identifier: MIT
from enum import Enum
from langchain.schema import (AIMessage, HumanMessage)
from src.chatbot.conversation import Conversation
from src.chatbot.prompt import Prompt
from src.models.llama_model import LlamaModel
from src.models.chatgpt_model import ChatGPT35Model, ChatGPT40Model
from src.scrapers.pdf_scraper import PDFScraper
from src.db.qdrant_db import QdrantDatabase

# This class represents a ChatBOT composed by:
# - chat history
# - the LLM model used to generate the AI answers
# - a Vector database to use to store user's data like, for example, a PDF.
# - the prompt
class ChatBOT:
    # Here the list of supported LLM models
    class Model(Enum):
        LLAMA = "llama"
        CHATGPT_3_5 = "chatgpt3.5"
        CHATGPT_4 = "chatgpt4"

    # By default, the ChatBOT uses the LLama 2 model and a temperature 0 that means answer more focused
    #  and choerent
    def __init__(self):
        self.model = LlamaModel(0.0)
        self.conversation = Conversation()
        self.db = QdrantDatabase()
        self.prompt = Prompt()

    # The user can change the default model with others supportedd.
    def set_model(self, model_type, temperature):
        if model_type == self.Model.LLAMA and not isinstance(self.model, LlamaModel):
            self.model = LlamaModel(temperature)
        elif model_type == self.Model.CHATGPT_3_5 and not isinstance(self.model, ChatGPT35Model):
            self.model = ChatGPT35Model(temperature)
        elif model_type == self.Model.CHATGPT_4 and not isinstance(self.model, ChatGPT40Model):
            self.model = ChatGPT40Model(temperature)
        #else:
        #    raise ValueError("Invalid model type")

    # Once the user insert the question, this method is called to generate the answer.
    # It leverages on the model get_answer method.
    def get_answer(self, question):
        context = self.db.get_context(question)
        prompt = self.prompt.get_prompt(context, question)
        self.conversation.add_message(HumanMessage(content=prompt))
        answer, cost = self.model.get_answer(self.conversation.get_messages())
        self.conversation.add_message(AIMessage(content=answer))
        self.conversation.add_cost(cost)
        return answer, cost

    def upload_pdf(self, pdf_file_path):
        pdf_scraper = PDFScraper(pdf_file_path)
        chunks = pdf_scraper.scrape()
        self.db.store(chunks, self.model.embeddings)

    # Return the chat history
    def get_chat_history(self):
        return self.conversation.get_messages()

    # Return the chat costs
    def get_chat_costs(self):
        return self.conversation.get_costs()

    # Clear the conversation
    def clear_conversation(self):
        self.conversation.clear()
