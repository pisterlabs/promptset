import os
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate

_ = load_dotenv(find_dotenv())  # read local .env file


# ---------------------------
# Generic Chain
#
# Text Completion with memory
#
# OPEN AI API - POST https://api.openai.com/v1/completions
# ---------------------------

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(
    ai_prefix="AI", human_prefix="Human", memory_key="chat_history"
)
llm = OpenAI(temperature=0)

# ------
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

output = llm_chain.predict(human_input="Hi there, my name is Sharon!")
print(output)

output = llm_chain.predict(
    human_input="What would be a good company name for a company that makes colorful socks?"
)
print(output)

output = llm_chain.predict(human_input="What is my name?")
print(output)

output = llm_chain.predict(human_input="Who are you in this conversation?")
print(output)
