from .Message import Message
from langchain.memory import ConversationBufferMemory
import uuid

class ConversationContext:
    context: str
    memory: ConversationBufferMemory
    messages: list[Message]

    def __init__(self, memory: ConversationBufferMemory, key: str = "", messages: list[Message] = []):
        self.memory = memory
        self.key = key if key != "" else str(uuid.uuid4())
        self.messages = messages

    def addMessage(self, message: Message):
        self.messages.append(message)

    def getContext(self):
        return self.context
    
class ConversationContextCache:
    contexts: dict[str, ConversationContext]

    def __init__(self):
        self.contexts = {}

    def addContext(self, context: ConversationContext):
        self.contexts[context.key] = context

    def getContext(self, key: str):
        return self.contexts[key]