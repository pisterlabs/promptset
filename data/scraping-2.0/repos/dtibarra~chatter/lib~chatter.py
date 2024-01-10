import openai
from collections import OrderedDict
from lib.schemas import Message, Convo


class Memory:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    # return a convo
    def get(self, thread_timestamp: int) -> int:
        if thread_timestamp not in self.cache:
            return False
        else:
            self.cache.move_to_end(thread_timestamp)
            return self.cache[thread_timestamp]

    # save a convo
    def push(self, thread_timestamp: int, message: Message) -> None:
        if thread_timestamp not in self.cache:
            self.cache[thread_timestamp] = Convo()
            self.cache[thread_timestamp].push(message)
        else:
            self.cache[thread_timestamp].push(message)
        self.cache.move_to_end(thread_timestamp)

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class Chatter:
    class Session:
        def __init__(self, chatter_instance, thread_timestamp, initialize_prompt=True):
            self.chatter_instance = chatter_instance
            self.thread_timestamp = thread_timestamp
            self.chatter_instance.memory.push(thread_timestamp, chatter_instance.prompt)

        async def chat(self, role, message):
            m = Message(role, message)
            self.chatter_instance.memory.push(self.thread_timestamp, m)
            response = await self.chatter_instance._chat(
                self.chatter_instance.memory.get(self.thread_timestamp)
            )
            response_message = Message(
                response["choices"][0]["message"]["role"],
                response["choices"][0]["message"]["content"],
            )
            self.chatter_instance.memory.push(self.thread_timestamp, response_message)
            return response_message

    def __init__(self, openapi_key, prompt="You are a chat bot."):
        openai.api_key = openapi_key
        self.prompt = Message("system", prompt)
        self.memory = Memory(10)
        self.id = None

    async def new_session(self, thread_timestamp):
        s = Chatter.Session(self, thread_timestamp)
        return s

    async def build_session(self, thread_timestamp):
        s = Chatter.Session(self, thread_timestamp, False)
        return s

    async def _chat(self, conversation):
        return await openai.ChatCompletion.acreate(
            model="gpt-4", messages=[m.asdict() for m in conversation.messages]
        )

    async def set_id(self, id):
        self.id = id

    async def get_id(self):
        return self.id
