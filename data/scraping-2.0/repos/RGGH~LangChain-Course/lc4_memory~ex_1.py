from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# initialize the LLM ~ large language model
llm = ChatOpenAI()

conversation_buf = ConversationChain(llm=llm, memory=ConversationBufferMemory())

res = conversation_buf("Good morning AI!")
print("\n", res)


print(conversation_buf.prompt.template)
# '''
# Current conversation:
# {history}
# Human: {input}
# AI:
# '''

# notice how history is empty {history} this is where conversational memory would appear

# If there is any history it gets passed into the model at the same time as your next question

"""By default, these chatbots don't have memory...
They treat each thing you say as a separate, new conversation, 
even if it's related to what you said before. 
But in certain cases, like when you're dealing with customer service bots 
or interactive storylines, it's important for the bot to remember what's been said earlier.

So, to make the chatbot remember, they have to add a "memory" feature to it. 
This feature lets the chatbot keep track of what you've said previously 
so that the conversation flows more naturally, just like how you'd remember 
what you and your friend talked about last time you met."""


"""./lc_chains/ex_4.py"""
