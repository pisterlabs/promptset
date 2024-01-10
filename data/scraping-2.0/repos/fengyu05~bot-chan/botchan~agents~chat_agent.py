from functools import cached_property
from typing import Optional

import structlog
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from botchan.agents.chat_agent_prompt import Prompt
from botchan.index.knowledge_index import (
    KNOWLEDGE_INDEX_RECOVERY_PROMPT_INPUT_VARIABLES,
    KNOWLEDGE_INDEX_RECOVERY_PROMPT_TEMPLATE,
    KnowledgeIndex,
)
from botchan.message_intent import MessageIntent
from botchan.settings import BOT_NAME, OPENAI_GPT_MODEL_ID
from botchan.slack.data_model import Message, MessageEvent
from botchan.slack.messages_fetcher import MessagesFetcher

logger = structlog.getLogger(__name__)


class MultiThreadChatAgent:
    def __init__(
        self, fetcher: MessagesFetcher, bot_user_id: str, knowledge_folder: str = None
    ) -> None:
        self.llm_chain_pool = {}  # keyed by thread_message_id
        self.bot_user_id = bot_user_id
        self.fetcher = fetcher
        self.knowledge_folder = knowledge_folder

    @cached_property
    def knowledge_index(self):
        return (
            self._load_knowledge_index(self.knowledge_folder)
            if self.knowledge_folder
            else None
        )

    def _new_llm_chain(self, intent: MessageIntent) -> LLMChain:
        prompt = Prompt.from_intent(intent)
        llm_chain = LLMChain(
            llm=OpenAI(
                model_name=OPENAI_GPT_MODEL_ID,
                temperature=0,
            ),
            prompt=PromptTemplate(
                input_variables=prompt.input_variables, template=prompt.template
            ),
            verbose=True,  # Turn this on if we need verbose logs for the prompt
            memory=ConversationBufferWindowMemory(
                k=prompt.memory_buffer,
                input_key=prompt.input_key,  # The key of the input variables that to be kept in memero
            ),
        )
        return llm_chain

    def _load_knowledge_index(self, knowledges_folder: str) -> KnowledgeIndex:
        return KnowledgeIndex.from_folder(knowledges_folder)

    def _locate_knowledge(self, question: str) -> str:
        assert self.knowledge_index is not None, "knowledge index is missing"
        meta_chain = LLMChain(
            llm=OpenAI(
                model_name=OPENAI_GPT_MODEL_ID,
                temperature=0,
            ),
            prompt=PromptTemplate(
                input_variables=KNOWLEDGE_INDEX_RECOVERY_PROMPT_INPUT_VARIABLES,
                template=KNOWLEDGE_INDEX_RECOVERY_PROMPT_TEMPLATE,
            ),
            verbose=False,
        )
        knowledge_file_answers = meta_chain.predict(
            question=question, index_mapping=self.knowledge_index.index_mapping_str
        )
        knowledge_files = knowledge_file_answers.split(",")
        return self.knowledge_index.learn_all_knowledge_flat(
            knowledge_files=knowledge_files
        )

    def _get_thread_messages(
        self, message_event: MessageEvent, mentioned_user_id: Optional[str] = None
    ) -> list[Message]:
        assert message_event.thread_ts, "thread_ts can't be non in _get_thread_messages"
        return self.fetcher.fetch(
            channel_id=message_event.channel,
            thread_ts=message_event.thread_ts,
            mentioned_user_id=mentioned_user_id,
        )

    def _get_llm_chain(
        self, message_event: MessageEvent, intent: MessageIntent
    ) -> LLMChain:
        cached_llm_id = f"{message_event.thread_message_id}|{intent.name.lower()}"

        if not cached_llm_id in self.llm_chain_pool:
            chatgpt_chain = self._new_llm_chain(intent)
            self.llm_chain_pool[cached_llm_id] = chatgpt_chain

            # if there is in a thread, load the previous context
            if not message_event.is_thread_root:
                messages = self._get_thread_messages(message_event)
                for message in messages:
                    if (
                        message.ts == message_event.ts
                    ):  # Do not need to reload the current message into memory
                        continue
                    if message.user == message_event.user:
                        chatgpt_chain.memory.chat_memory.add_user_message(message.text)
                    if message.user == self.bot_user_id:
                        chatgpt_chain.memory.chat_memory.add_ai_message(message.text)

        return self.llm_chain_pool[cached_llm_id]

    def run(self, message_event: MessageEvent, message_intent: MessageIntent) -> str:
        chatgpt_chain = self._get_llm_chain(message_event, message_intent)

        if message_intent == MessageIntent.TECH_CHAT:
            return chatgpt_chain.predict(
                human_input=message_event.text,
                bot_name=BOT_NAME,
                philosophy_style="modern",
                knowledge=self.knowledge_index.locate_knowledge(message_event.text)
                if self.knowledge_index
                else "",
            )
        else:
            return chatgpt_chain.predict(
                human_input=message_event.text,
                bot_name=BOT_NAME,
                philosophy_style="modern",
            )
