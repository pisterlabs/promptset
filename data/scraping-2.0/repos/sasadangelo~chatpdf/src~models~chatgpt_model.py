# ChatGPTModel - ChatGPT models
#
# This class represents a generic ChatGPT model.
#
# Copyright (C) 2023 Salvatore D'Angelo
# Maintainer: Salvatore D'Angelo sasadangelo@gmail.com
#
# SPDX-License-Identifier: MIT
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from src.models.base_model import Model

# This class represents a generic ChatGPT model. It uses OpenAPI to interaact with ChatGPT.
class ChatGPTModel(Model):
    # Initialize the model with the name and temperature.
    def __init__(self, temperature, model_name):
        super().__init__(temperature, model_name, OpenAIEmbeddings())
        self.model = ChatOpenAI(temperature=temperature, model_name=model_name)

    # Return the model's answer to the user's question.
    def get_answer(self, messages) -> tuple[str, float]:
        with get_openai_callback() as cb:
            answer = self.model(messages)
        return answer.content, cb.total_cost

# This class represents the ChatGPT 3.5 model.
class ChatGPT35Model(ChatGPTModel):
    def __init__(self, temperature):
        super().__init__(temperature, "gpt-3.5-turbo-0613")

# This class represents the ChatGPT 4.0 model.
class ChatGPT40Model(ChatGPTModel):
    def __init__(self, temperature):
        super().__init__(temperature, "gpt-4")
