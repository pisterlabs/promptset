import spacy
from langchain import OpenAI, ConversationChain
from langchain.schema import BaseMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema.messages import BaseMessage, get_buffer_string
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Any
nlp = spacy.load("en_core_web_lg")
from langchain.memory.utils import get_prompt_input_key
from langchain.schema.messages import get_buffer_string,BaseMessage, HumanMessage, AIMessage


class _ConversationBufferMemory(BaseChatMemory):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        if self.return_messages:
            return self.chat_memory.messages
        else:
            return get_buffer_string(
                self.chat_memory.messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )

    def del_msg(self, message: BaseMessage) -> None:
        try: 
            self.chat_memory.messages.remove(message)
        except ValueError as e:
            print(e) 
        
    def export_dict(self,) -> List[Dict[str, str]]:
        message = []
        for i in self.chat_memory.messages:
            meme :dict[str, str] = {}
            author: str = ''
            if type(i) == HumanMessage:
                author = 'H'
            if type(i) == AIMessage:
                author = "A"
            meme[author]=i.content
            message.append(meme)
        return message

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}


