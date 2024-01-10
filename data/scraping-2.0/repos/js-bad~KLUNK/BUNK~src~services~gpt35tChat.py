import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from ..models.ConversationContext import ConversationContext, ConversationContextCache
from ..models.Interaction import Interaction
from ..models.Message import Message, MessageType

# TODO -- turn this whole file into a class
# TODO -- preload contexts from database
conversationContextCache = ConversationContextCache()

def chat(interaction: Interaction):
    llm = ChatOpenAI(temperature=1, verbose=True)
    waifu_template = PromptTemplate(
        input_variables = ['proompt'],
        template='''
        At the end of this prompt, there will be a message from a user, which will be wrapped in quotation marks. 
        Pretend that you are a kawaii anime waifu. Come up with a cute nickname for the user, and respond to their message in-character.
        \"{proompt}\"
        '''
    )

    if (interaction.conversationContext is None):
        waifu_memory = ConversationBufferMemory(input_key='proompt', memory_key='chat_history')
        conversationContext = ConversationContext(memory=waifu_memory)
        interaction.conversationContext = conversationContext.key
        conversationContextCache.addContext(conversationContext)
    else:
        conversationContext = conversationContextCache.getContext(interaction.conversationContext)
        # TODO -- error handling for invalid context key
        waifu_memory = conversationContext.memory

    waifu_chain = LLMChain(llm=llm, prompt=waifu_template, verbose=True, output_key='response', memory=waifu_memory)
    
    responseStr = waifu_chain.run({'proompt': interaction.proompt.content})
    llmResponse = Message(content=responseStr, type=MessageType.BOT, conversationContext=interaction.proompt.conversationContext)

    conversationContextCache.getContext(interaction.conversationContext).addMessage(llmResponse)
    interaction.response = llmResponse

    # TODO -- Save context to database
    
    return interaction