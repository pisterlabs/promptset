# LlamaModel - LLama 2 Model
#
# This class represents the LLama 2 model. It is used to create the prompt and the answer to the user input using
# the LLama 2 model.
#
# Copyright (C) 2023 Salvatore D'Angelo
# Maintainer: Salvatore D'Angelo sasadangelo@gmail.com
#
# SPDX-License-Identifier: MIT
from typing import List, Union
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.embeddings import LlamaCppEmbeddings
from src.models.base_model import Model

# This class represents the LLamaa 2 model used to generate the user's reply.
class LlamaModel(Model):
    # This method initialize the LLama 2 model calling the superclass contructor and passing
    # the model name (llama-2-7b-chat.ggmlv3.q2_K) and the temperature. The temperature is a
    # model's configuration parameter ranging from 0 to 1 to provide coherent vs creative answers.
    def __init__(self, temperature):
        super().__init__(temperature, "llama-2-7b-chat.ggmlv3.q2_K", LlamaCppEmbeddings(model_path=f"./models/llama-2-7b-chat.ggmlv3.q2_K.bin"))
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.model = LlamaCpp(
            model_path=f"./models/{self.model_name}.bin",
            input={"temperature": temperature,
                   "max_length": 2048,
                   "top_p": 1
                   },
            n_ctx = 2048,
            callback_manager=callback_manager,
            verbose=False,  # True
        )

    def get_answer(self, messages) -> tuple[str, float]:
        return self.model(self.__llama_v2_prompt(self.__convert_langchainschema_to_dict(messages))), 0.0

    # This method creates the prompt for the model. It uses a set of messages formed by:
    # - system message
    # - chat history
    # - the question
    def __llama_v2_prompt(self, messages: List[dict]) -> str:
        """
        Convert the messages in list of dictionary format to Llama2 compliant format.
        """
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        BOS, EOS = "<s>", "</s>"
        DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        if messages[0]["role"] != "system":
            messages = [
                {
                    "role": "system",
                    "content": DEFAULT_SYSTEM_PROMPT,
                }
            ] + messages
        messages = [
            {
                "role": messages[1]["role"],
                "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
            }
        ] + messages[2:]

        messages_list = [
            f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
            for prompt, answer in zip(messages[::2], messages[1::2])
        ]
        messages_list.append(
            f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")
        return "".join(messages_list)

    # This method works at support of the get_answer method.
    def __convert_langchainschema_to_dict(self,
            messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
            -> List[dict]:
        """
        Convert the chain of chat messages in list of langchain.schema format to
        list of dictionary format.
        """
        return [{"role": self.__find_role(message),
                "content": message.content
                } for message in messages]

    # Depending on message type in input it returns:
    # - system
    # - user
    # - aassistant
    def __find_role(self, message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
        """
        Identify role name from langchain.schema object.
        """
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        raise TypeError("Unknown message type.")
