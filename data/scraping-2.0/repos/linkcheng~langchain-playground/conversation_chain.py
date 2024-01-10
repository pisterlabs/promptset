import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (
    ConversationBufferMemory, 
    ConversationSummaryMemory
)
from langchain.callbacks import get_openai_callback

load_dotenv()


# 相关性聊天

def load_chain():
    key = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(openai_api_key=key, temperature=0, model_name='text-davinci-003')
    chain = ConversationChain(llm=llm, memory=ConversationBufferMemory())
    return chain


def load_summary_chain():
    key = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(openai_api_key=key, temperature=0, model_name='text-davinci-003')
    chain = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm))
    return chain

def track_tokens_usage(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f"{cb}")
    return result

# main
if __name__ == '__main__':
    chain = load_chain()
    
    print(f"{chain.prompt.template=}")
    ret = track_tokens_usage(chain, 'what is blockchain?')
    print(ret)
    
    print(f"{chain.memory.buffer=}")
    ret = track_tokens_usage(chain, 'what did I just ask you?')
    print(ret)
