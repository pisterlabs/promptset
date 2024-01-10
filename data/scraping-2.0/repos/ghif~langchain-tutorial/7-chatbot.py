from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate, FewShotPromptTemplate, SemanticSimilarityExampleSelector
from langchain.chains import LLMChain, RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

import chainlit as cl
import json

"""
Load OpenAI API key 
"""
_ = load_dotenv(find_dotenv())
prefix = """You are a knowledgeable customer service from Pusat Bantuan Merdeka Belajar Kampus Merdeka (MBKM).
Use the context below to answer various questions from users.
If you don't know the answer, just say I don't know. Don't make up an answer.
The answer given must always be in Indonesian language with a friendly tone.

The example response you give is as follows:

```
Terima kasih telah menghubungi Pusat Bantun Kampus Merdeka.
....


Salam hangat,
Tim Kampus Merdeka
```

Here are some examples of conversations between users and customer service to be your references:
"""

suffix = """
Question: {query}
Answer: 
"""
embeddings = OpenAIEmbeddings()

# load few shot conversation examples
examples = json.load(open("chat_samples.json", "r"))


# Initialize chat
@cl.on_chat_start
def init():
    """
    Model
    """
    chat_llm = ChatOpenAI(
        temperature=0.3,
        streaming=True
    )

    """
    Chain
    """
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="Question: {question}\nAnswer: {answer}",
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples, 
        embeddings,
        FAISS, 
        k=5 # k-nearest neighbors
    )

    fewshot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
    )

    prompt = ChatPromptTemplate.from_template(prefix)
    chain = LLMChain(
        llm=chat_llm,
        # prompt=prompt,
        prompt=fewshot_prompt,
        verbose=True
    )
    
    cl.user_session.set("chain", chain)
    

@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")

    # Infer from the chain
    outputs = await chain.acall(
        message,
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

    # Post-processing (if any)
    res = outputs["text"]

    # Send the response 
    await cl.Message(
        content=res
    ).send()