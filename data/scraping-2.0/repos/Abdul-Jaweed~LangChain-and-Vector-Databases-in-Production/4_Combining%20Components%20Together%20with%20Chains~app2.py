## Conversational Chain (Memory)

# Depending on the application, memory is the next component that will complete a chain. LangChain provides a ConversationalChain to track previous prompts and responses using the ConversationalBufferMemory class.

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import PromptTemplate, OpenAI, LLMChain

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0
)

output_parser = CommaSeparatedListOutputParser()

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

conversation.predict(input="List all possible words a substitute for 'artificial' as comma separated.")


# Now, we can ask it to return the following four replacement words. It uses the memory to find the next options.

conversation.predict(input="And the next 4?")