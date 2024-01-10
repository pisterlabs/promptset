from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage

load_dotenv(find_dotenv())


class ConversationMemoryLoader:
    def __init__(self, memory, data):
        self._data = data
        self._memory = memory
        self._messages: list[BaseMessage] = []

    @property
    def messages(self):
        return self._messages

    def _create_messages(self):
        role_class_map = {
            "ai": AIMessage,
            "human": HumanMessage,
            "system": SystemMessage
        }
        self._messages = [role_class_map[message["role"]](content=message["content"]) for message in self._data]
        return self

    def _update_memory(self):
        # Limpa a memória.
        self._memory.load_memory_variables({})
        # Processa as mensagens
        last_human_msg = None
        for msg in self._messages:
            if isinstance(msg, HumanMessage):
                last_human_msg = msg
            elif isinstance(msg, AIMessage) and last_human_msg is not None:
                self._memory.save_context(
                    {"input": last_human_msg.content},
                    {"output": msg.content}
                )
                last_human_msg = None
        return self

    def load(self):
        self._create_messages()._update_memory()
        return self


class LLMConnection:

    def __init__(self, data):
        self._data = data
        self._messages: list[BaseMessage] = []
        self._memory = ConversationBufferMemory(return_messages=True, ai_prefix="AI Assistant")
        self._prompt = None
        self._output = ""

    @property
    def output(self):
        return self._output

    def _create_prompt(self):
        system_message = self._messages.pop(0)
        prompt = system_message.content + """
        
        Conversa atual:
        {history}
        
        AI Assistant: 
        {input}
        """
        self._prompt = PromptTemplate(input_variables=["history", "input"], template=prompt)
        return self

    def send(self):
        data = self._data["conversation"]["messages"]
        self._messages = ConversationMemoryLoader(self._memory, data).load().messages
        self._create_prompt()
        message = self._messages[-1]
        llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")
        chain = ConversationChain(
            llm=llm,
            memory=self._memory,
            prompt=self._prompt,
        )
        self._output = chain.predict(input=message.content)
        return self
