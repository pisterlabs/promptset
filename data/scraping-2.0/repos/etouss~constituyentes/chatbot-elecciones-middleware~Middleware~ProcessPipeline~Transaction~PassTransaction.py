from ..Conversation import Conversation
from ..GenericTransaction import GenericTransaction
from ..TransactionChainChat import TransactionChainChat

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

class PassTransaction(GenericTransaction):
    def __init__(self,conversation:Conversation) -> None:
        super().__init__(name = 'pass',conversation = conversation)

    def process_input(self, input:str) -> str:
        txc = TransactionChainChat(self.tx_id)


        #output = chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
        messages = [
            SystemMessage(content="You are a helpful assistant that translates Spanish to English."),
            HumanMessage(content="Translate this sentence from Spanish to English. "+input)
        ]

        return txc.process_input(messages)
        