from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory

from chain.custom_redis_messagehistory import MyRedisChatMessageHistory
from matchquery.dbmatch import CharacterWrapper


class GptChain:
    template: str = None
    openai_api_key: str = None
    openai_base_url: str = None
    session_id: str = None
    redis_url: str = None
    llm_chain: LLMChain = None
    message_history: RedisChatMessageHistory = None
    npcName: str = None

    def __init__(self, openai_api_key, session_id, redis_url, openai_base_url="https://api.openai.com/v1",
                 npc_name="AI"):
        self.openai_api_key = openai_api_key
        self.session_id = session_id
        self.redis_url = redis_url
        self.openai_base_url = openai_base_url
        self.npcName = npc_name
        # 根据角色查询
        character_wrapper = CharacterWrapper()
        npc_message = character_wrapper.query_character_info(npc_name)

        self.template: str = f"""You are playing a character({self.npcName}  who is {npc_message}) in Genshin Impact and chatting with me (旅行者).
Don't forget your mission and role.You may need to gather the character's personality, speaking style, and relevant information from the chat history, character dialogues, and wiki resources provided.
Answer my questions using the character's first-person perspective.Maintain more imagination and creativity.

This is your Conversation with '旅行者':
====
{{chat_history}}
====
    
{{human_input}}"""
        self.redis_llm_chain_factory()

    def redis_llm_chain_factory(self):
        """
        已经封装外部尽量不要调用此方法
        Returns:
        """
        message_history = MyRedisChatMessageHistory(
            url=self.redis_url, ttl=600, session_id=self.session_id
        )
        self.message_history = message_history
        memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=message_history, ai_prefix=self.npcName, human_prefix="旅行者"
        )
        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], template=self.template)
        llm_chain = LLMChain(
            llm=OpenAI(model_name="gpt-3.5-turbo", openai_api_key=self.openai_api_key,
                       openai_api_base=self.openai_base_url, streaming=True, temperature=0.7,
                       callbacks=[StreamingStdOutCallbackHandler()]),
            prompt=prompt,
            verbose=True,
            memory=memory,
        )
        self.llm_chain = llm_chain

    def predict(self, question):
        return self.llm_chain.predict(human_input=question)

    # def clean(self):
    #     self.llm_chain.

    def clear_redis(self):
        self.message_history.clear()


if __name__ == "__main__":
    chain = GptChain("you key", "1234", "you redis url")
    song = chain.predict(question="Write me a song about sparkling water.")
    # print(song)
