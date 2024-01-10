import logging

from langchain import PromptTemplate
from langchain.chains import LLMChain as llm_chain
from langchain.memory import ConversationTokenBufferMemory

from shared.selector import get_llm, get_chat_model
from utilities.token_helper import num_tokens_from_string

from ai.configurations.llm_chain_configuration import LLMChainConfiguration
from ai.abstract_ai import AbstractAI
from ai.ai_result import AIResult


class LLMChain(AbstractAI):
    def configure(self, json_args) -> None:
        self.configuration = LLMChainConfiguration(json_args)
        if self.configuration.chat_model:
            llm = get_chat_model(
                self.configuration.run_locally, float(self.configuration.ai_temp)
            )
        else:
            llm = get_llm(
                self.configuration.run_locally,
                local_model_path=self.configuration.model,
                ai_temp=float(self.configuration.ai_temp),
                max_tokens=-1,
            )

        memory = self._get_memory(llm) if self.configuration.use_memory else None

        if self.configuration.prompt:
            if "inputs" in self.configuration.prompt:
                prompt = PromptTemplate.from_template(self.configuration.prompt)
            else:
                raise Exception("Prompt must contain 'inputs' key")
        else:
            prompt = PromptTemplate.from_template("{inputs}")

        self.chain = llm_chain(
            llm=llm, memory=memory, verbose=self.configuration.verbose, prompt=prompt
        )

    def _get_memory(self, llm):
        memory = ConversationTokenBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            input_key="inputs",
            output_key="text",
        )

        return memory

    def query(self, input):
        num_tokens = 0

        if isinstance(input, str):
            # If the input is a single string
            num_tokens = num_tokens_from_string(input)
        elif isinstance(input, list):
            # If the input is a list of strings
            for string in input:
                num_tokens += num_tokens_from_string(string)

        logging.debug(f"LLMChain query has {num_tokens} tokens")

        result = self.chain(inputs=input)

        ai_results = AIResult(result, result["text"])

        return ai_results
