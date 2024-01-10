from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Protocol

from langchain.chains import LLMChain
from llama_cpp import suppress_stdout_stderr


class LanguageModel:
    def __init__(self, llm_chain: LLMChain):
        self.llm_chain = llm_chain

    def awnser(self, prompt: str) -> str:
        if self.llm_chain:
            awnser = self.llm_chain.predict(user_input=prompt)
        else:
            raise RuntimeError("Failed to load llm model")

        return awnser

    def messages(self):
        return self.llm_chain.memory.chat_memory.messages

    def load_memory(self, memory_file_location: str):
        with open(memory_file_location, "rb") as memory_file:
            with suppress_stdout_stderr():
                memory = pickle.load(memory_file)
        self.llm_chain.memory = memory

    def save_memory(self, memory_file_location: str):
        with open(memory_file_location, "wb") as memory_file:
            pickle.dump(self.llm_chain.memory, memory_file)