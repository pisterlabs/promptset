from langchain import OpenAI, ConversationChain
import os
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

llm = OpenAI(temperature = 0)

conversation = ConversationChain(llm=llm, verbose=True)

conversation.predict(input="Hello World!")

conversation.predict(input="What is the first thing I said to you?")

conversation.predict(input="What is an alternative phrase for the first thing I said to you?")