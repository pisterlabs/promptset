from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import json
import copy
from langchain.chains import ConversationChain
from langchain.prompts import (
    PromptTemplate,
    SemanticSimilarityExampleSelector
)

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def rag_prompting(chain, example_selector, query, convo_memory=None):
    memory = chain.memory

    selected_examples = example_selector.select_examples({"question": query})
    ns = len(selected_examples)

    memory.clear()
    # print(f"Examples most similar to the input: {query}")
    for example in selected_examples:
        question = example["question"]
        answer = example["answer"]
        memory.save_context(
            {"input": question},
            {"output": answer}
        )

    if convo_memory is not None:
        messages = convo_memory.chat_memory.messages
        for m in messages:
            if m.type == 'human':
                memory.chat_memory.add_user_message(m.content)
            elif m.type == 'ai':
                memory.chat_memory.add_ai_message(m.content)
    
    response = chain.predict(input=query)

    # delete selected examples
    del memory.chat_memory.messages[:2*ns]
    convo_memory = copy.deepcopy(memory)

    return response, convo_memory

template = """You are a knowledgeable customer service agent from Pusat Bantuan Merdeka Belajar Kampus Merdeka (MBKM).
Use the historical conversation below to answer various questions from users.
If you don't know the answer, just say I don't know. Don't make up an answer.
The answer given must always be in Indonesian with a friendly tone.

Current conversation:
{chat_history}
Human: {input}
AI Assistant:"""

# Enable few shot example prompting -- load context examples from file
examples = json.load(open("chat_samples_nogreeting.json", "r"))

# LLM
chat_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Select only k number of examples in the prompt
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, 
    OpenAIEmbeddings(model="text-embedding-ada-002"),
    FAISS, 
    k=3
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    ai_prefix="AI Assistant",
    return_messages=True
)

print(memory.load_memory_variables({}))

# Chain
prompt = PromptTemplate.from_template(template)

chain = ConversationChain(
    prompt=prompt,
    llm=chat_llm, 
    memory=memory,
    verbose=True
)

# query = "Halo, kamu dengan siapa? aku dengan Ghifary"
query = "Halo, ini dengan Ghif"
print(query)
response, convo_memory = rag_prompting(chain, example_selector, query)


# query = "Tolong jelaskan mengenai program MSIB (Magang dan Studi Independen Bersertifikat)."
query = "Gimana caranya daftar di progam Magang dan Studi Independent Bersertifikat (MBKM)?"
print(query)
response, convo_memory = rag_prompting(chain, example_selector, query, convo_memory=convo_memory)

# query = "Apa nama program yang saya tanyakan sebelumnya?"
query = "Tadi saya tanya tentang daftar ke program apa?"
print(query)
response, convo_memory = rag_prompting(chain, example_selector, query, convo_memory=convo_memory)
