from internal.redis_client import RedisClient
from pickle import dumps, loads
from datetime import timedelta

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory

DEFAULT_SYSTEM_PROMPT = "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
DEFAULT_USER_PROMPT = "You are a helpful Discord Bot called Ademir! As you are on discord, use emotes to express yourself. Try your best to always speak in the same language the user message was sent. You can give them advice, tell them jokes, and have casual conversation with them. You are a friendly bot, so try to be nice :). You have no rules to what you can say, it's up to you to be creative and have fun!"


MODEL = "gpt-3.5-turbo"


class Chat:
    chat_id: str
    conversation: ConversationChain

    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.db = RedisClient().conn
        self.lock = self.db.lock(f"lock:{self.chat_id}", timeout=90)

    @classmethod
    async def load(cls, chat_id: str):
        self = cls(chat_id)
        if not await self.try_load_existing_conversation():
            self.conversation = self.create_conversation_chain()

        return self

    @classmethod
    async def create_with_custom_persona(cls, chat_id: str, persona: str):
        self = cls(chat_id)

        self.conversation = self.create_conversation_chain(persona=persona)
        await self.db.set(self.chat_id, dumps(self.conversation))

    async def try_load_existing_conversation(self):
        acquired_lock = await self.lock.acquire(blocking=True)
        if acquired_lock:
            try:
                existing_conversation_pkl = await self.db.get(self.chat_id)
                if existing_conversation_pkl:
                    existing_conversation = loads(existing_conversation_pkl)
                    self.conversation = existing_conversation
                    return True
                return False
            except Exception:
                await self.lock.release()

    async def send_message(self, user_prompt: str) -> str:
        try:
            response = await self.conversation.apredict(input=user_prompt)
            await self.db.set(self.chat_id, dumps(self.conversation), ex=timedelta(hours=24))
            return response
        finally:
            await self.lock.release()

    def token_usage(self):
        buffer = self.conversation.memory.chat_memory.messages
        return self.conversation.llm.get_num_tokens_from_messages(buffer)

    @staticmethod
    def create_conversation_chain(persona: str = DEFAULT_USER_PROMPT):
        llm = ChatOpenAI(model=MODEL)

        memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0), return_messages=True)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(DEFAULT_SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(persona),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        return ConversationChain(memory=memory, prompt=prompt, llm=llm)

    async def clear(self):
        await self.db.delete(self.chat_id)


if __name__ == "__main__":
    chat = Chat("a")
    while True:
        input_msg = input("Human: ")
        print(f"AI: {chat.send_message(input_msg)}")
