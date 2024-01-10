import os
from typing import List, Optional

from langchain import ConversationChain, PromptTemplate
from langchain.chains.base import Chain
from langchain.memory import ConversationSummaryBufferMemory

from app_modules.llm_inference import LLMInference


def get_llama_2_prompt_template():
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    instruction = "Chat History:\n\n{history} \n\nUser: {input}"
    system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. Read the chat history to get context"
    # system_prompt = """\
    # You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. \n\nDo not output any emotional expression. Read the chat history to get context.\
    # """

    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


class ChatChain(LLMInference):
    def __init__(self, llm_loader):
        super().__init__(llm_loader)

    def create_chain(self, inputs) -> Chain:
        template = (
            get_llama_2_prompt_template()
            if os.environ.get("USE_LLAMA_2_PROMPT_TEMPLATE") == "true"
            else """You are a chatbot having a conversation with a human.
{history}
Human: {input}
Chatbot:"""
        )

        print(f"template: {template}")

        prompt = PromptTemplate(input_variables=["history", "input"], template=template)

        memory = ConversationSummaryBufferMemory(
            llm=self.llm_loader.llm, max_token_limit=1024, return_messages=True
        )

        llm_chain = ConversationChain(
            llm=self.llm_loader.llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

        return llm_chain

    def run_chain(self, chain, inputs, callbacks: Optional[List] = []):
        return chain({"input": inputs["question"]}, callbacks)
