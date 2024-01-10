from typing import Annotated, Optional

from fastapi import Depends, Header
from langchain.chains.base import Chain
from langchain_community.llms.huggingface_text_gen_inference import (
    HuggingFaceTextGenInference,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from chatbot.callbacks import TracingLLMCallbackHandler
from chatbot.chains import LLMConvChain
from chatbot.config import settings
from chatbot.history import ChatbotMessageHistory
from chatbot.memory import ChatbotMemory
from chatbot.prompts.chatml import AI_SUFFIX, HUMAN_PREFIX, ChatMLPromptTemplate


def UserIdHeader(alias: Optional[str] = None, **kwargs):
    if alias is None:
        alias = settings.user_id_header
    return Header(alias=alias, **kwargs)


def UsernameHeader(alias: Optional[str] = None, **kwargs):
    if alias is None:
        alias = "X-Forwarded-Preferred-Username"
    return Header(alias=alias, **kwargs)


def EmailHeader(alias: Optional[str] = None, **kwargs):
    if alias is None:
        alias = "X-Forwarded-Email"
    return Header(alias=alias, **kwargs)


def MessageHistory() -> BaseChatMessageHistory:
    return ChatbotMessageHistory(
        url=str(settings.redis_om_url),
        key_prefix="chatbot:messages:",
        session_id="sid",  # a fake session id as it is required
    )


def ChatMemory(
    history: Annotated[BaseChatMessageHistory, Depends(MessageHistory)]
) -> BaseMemory:
    return ChatbotMemory(
        memory_key="history",
        input_key="input",
        history=history,
        return_messages=True,
    )


def Llm() -> BaseLLM:
    return HuggingFaceTextGenInference(
        inference_server_url=str(settings.inference_server_url),
        max_new_tokens=1024,
        stop_sequences=[AI_SUFFIX, HUMAN_PREFIX],
        streaming=True,
        callbacks=[TracingLLMCallbackHandler()],
    )


def ConvChain(
    llm: Annotated[BaseLLM, Depends(Llm)],
    memory: Annotated[BaseMemory, Depends(ChatMemory)],
) -> Chain:
    system_prompt = PromptTemplate(
        template="""You are Rei, the ideal assistant dedicated to assisting users effectively.
Knowledge cutoff: 2023-10-01
Current date: {date}
Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.""",
        input_variables=["date"],
    )
    messages = [
        SystemMessagePromptTemplate(prompt=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
    tmpl = ChatMLPromptTemplate(input_variables=["date", "input"], messages=messages)
    return LLMConvChain(
        user_input_variable="input",
        llm=llm,
        prompt=tmpl,
        verbose=False,
        memory=memory,
    )
