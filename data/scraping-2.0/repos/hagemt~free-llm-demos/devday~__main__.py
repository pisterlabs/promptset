#!/usr/bin/env python3
import logging, os, pinecone, sys  # noqa: E401
from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter as RCTS
from langchain.vectorstores import Pinecone

def chat(*args, **kwargs) -> str | list[str | dict]:
    gpt_model_name = os.getenv("GPT_MODEL_NAME", "gpt-3.5-turbo")
    llm = ChatOpenAI(model=gpt_model_name, temperature=0.9, **kwargs)
    res = llm([
        SystemMessage(content="You are an expert data scientist"),
        HumanMessage(content="Write a Python script that trains a neural network on simulated data "),
        *args
    ])
    # https://api.python.langchain.com/en/latest/schema/langchain.schema.messages.AIMessage.html
    return res.content or "OK"

def demo(concept: str, verbose=True):
    #print(chat()) # basic AIMessage example
    llm = OpenAI(model_name=os.getenv("GPT_MODEL_NAME", "text-davinci-003"))
    prompt1 = PromptTemplate(input_variables=["ml_concept"], template="""
You are an expert data scientist with an expertise in building deep learning models.
Explain the concept of {ml_concept} in a couple of lines
""")
    prompt2 = PromptTemplate(input_variables=["ml_concept"], template="""
Explain {ml_concept} to me like I'm five
""")
    #print(llm(prompt.format(concept="autoencoder")))
    links = [LLMChain(llm=llm, prompt=p) for p in (prompt1, prompt2)]
    chain = SimpleSequentialChain(chains=links, verbose=verbose)
    return chain.run(concept)

def init() -> None:
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),
    )

def main(*args) -> None:
    # if 1st arg refers to known sub-command, do that:
    match sub := args[0] if len(args) > 0 else "main":
        case "beta" | "talk":
            import importlib
            mod = importlib.import_module(sub)
            mod.main(*args[1:])
            return

    out = demo("autoencoder")
    splitter = RCTS(chunk_size=100, chunk_overlap=0)
    texts = splitter.create_documents([out])
    embed = OpenAIEmbeddings(tiktoken_model_name="ada")

    init()
    search = Pinecone.from_documents(texts, embed, index_name="lc-quickstart")
    result = search.similarity_search("What is magical about an autoencoder?")
    print(result)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv(find_dotenv())
    main(*sys.argv[1:])
