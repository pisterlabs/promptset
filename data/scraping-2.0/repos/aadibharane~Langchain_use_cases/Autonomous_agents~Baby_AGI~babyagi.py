'''
BabyAGI is an AI agent that can generate and pretend to execute tasks based on a given objective.

This guide will help you understand the components to create your own recursive agents.

Although BabyAGI uses specific vectorstores/model providers (Pinecone, OpenAI), one of the benefits of implementing it with LangChain 
is that you can easily swap those out for different options. In this implementation we use a FAISS vectorstore (because it runs locally and is free).
'''
#Install and Import Required Modules
'''
import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI

#Connect to the Vector Store
#Depending on what vectorstore you use, this step may look different.
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# Define your embedding model
def baby_agi():
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty

    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    #Run the BabyAGI
    #Now it’s time to create the BabyAGI controller and watch it try to accomplish your objective.

    #OBJECTIVE = "Write a weather report for SF today"
    #OBJECTIVE = "Be an reporter for IPL cricket most recent match"
    #OBJECTIVE = "How to become java developer"
    OBJECTIVE= "How to make butter chicken"


    llm = OpenAI(temperature=0)

    # Logging of LLMChains
    verbose = False
    # If None, will keep on going forever
    max_iterations: Optional[int] = 3
    baby_agi = BabyAGI.from_llm(
        llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
    )

    res=baby_agi({"objective": OBJECTIVE})
    print(res)
baby_agi()
'''

# import os
# import tkinter as tk
# from tkinter import messagebox
# from collections import deque
# from typing import Dict, List, Optional, Any

# from langchain import LLMChain, OpenAI, PromptTemplate
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.llms import BaseLLM
# from langchain.vectorstores.base import VectorStore
# from pydantic import BaseModel, Field
# from langchain.chains.base import Chain
# from langchain.experimental import BabyAGI
# from langchain.vectorstores import FAISS
# from langchain.docstore import InMemoryDocstore
# import faiss

# os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# def baby_agi():
#     embeddings_model = OpenAIEmbeddings()
#     embedding_size = 1536
#     index = faiss.IndexFlatL2(embedding_size)
#     vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

#     OBJECTIVE = input("enter a response: ")#"How to make butter chicken"

#     llm = OpenAI(temperature=0)
#     verbose = False
#     max_iterations: Optional[int] = 3
#     baby_agi = BabyAGI.from_llm(
#         llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
#     )

#     res = baby_agi({"objective": OBJECTIVE})
#     return res

# def run_baby_agi():
#     response = baby_agi()
#     messagebox.showinfo("Baby AGI Response", response)

# window = tk.Tk()
# window.title("Baby AGI Response")

# button = tk.Button(window, text="Run Baby AGI", command=run_baby_agi)
# button.pack()

# window.mainloop()
'''
import os
import tkinter as tk
from tkinter import messagebox
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

def run_baby_agi():
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    OBJECTIVE = "How to make butter chicken"

    llm = OpenAI(temperature=0)
    verbose = False
    max_iterations: Optional[int] = 3
    baby_agi = BabyAGI.from_llm(
        llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
    )

    res = baby_agi({"objective": OBJECTIVE})
    messagebox.showinfo("Baby AGI Response", res)

window = tk.Tk()
window.title("Baby AGI Response")

button = tk.Button(window, text="Run Baby AGI", command=run_baby_agi)
button.pack()

window.mainloop()
'''
# import os
# import tkinter as tk
# from tkinter import messagebox
# from collections import deque
# from typing import Dict, List, Optional, Any

# from langchain import LLMChain, OpenAI, PromptTemplate
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.llms import BaseLLM
# from langchain.vectorstores.base import VectorStore
# from pydantic import BaseModel, Field
# from langchain.chains.base import Chain
# from langchain.experimental import BabyAGI
# from langchain.vectorstores import FAISS
# from langchain.docstore import InMemoryDocstore
# import faiss

# os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# def run_baby_agi():
#     embeddings_model = OpenAIEmbeddings()
#     embedding_size = 1536
#     index = faiss.IndexFlatL2(embedding_size)
#     vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

#     OBJECTIVE =input("Enter your prompt here: ")  #"How to make butter chicken"

#     llm = OpenAI(temperature=0)
#     verbose = False
#     max_iterations = 3
#     baby_agi = BabyAGI.from_llm(
#         llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
#     )

#     res = baby_agi({"objective": OBJECTIVE})
#     response_label.config(text=res)

# window = tk.Tk()
# window.title("Baby AGI Response")

# button = tk.Button(window, text="Run Baby AGI", command=run_baby_agi)
# button.pack()

# response_label = tk.Label(window, text="")
# response_label.pack()

# window.mainloop()
'''
import os
import tkinter as tk
from tkinter import messagebox
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

def run_baby_agi():
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    OBJECTIVE = input("Enter your prompt here: ")  # "How to make butter chicken"

    llm = OpenAI(temperature=0)
    verbose = False
    max_iterations = 3
    baby_agi = BabyAGI.from_llm(
        llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
    )

    res = baby_agi({"objective": OBJECTIVE})
    response_label.config(text=res)

# window = tk.Tk()
# window.title("Baby AGI Response")

# button = tk.Button(window, text="Run Baby AGI", command=run_baby_agi)
# button.pack()

# response_label = tk.Label(window, text="", wraplength=400)
# response_label.pack()

# window.mainloop()

root = tk.Tk()
root.title("ChatGPT Clone")

# Create an entry field for input
input_entry = tk.Entry(root, width=50)
input_entry.pack()

def button_click():
    response = run_baby_agi()
    response_label.config(text=response)

# Create a label to display the response
response_label = tk.Label(root, text="", wraplength=400)
response_label.pack()

# Create a button to trigger the backend function
button = tk.Button(root, text="Run", command=button_click)
button.pack()

# Run the Tkinter event loop
root.mainloop()

'''

import os
import tkinter as tk
from tkinter import messagebox
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

def chatgpt_clone():
    template = """Assistant is a large language model trained by OpenAI.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    {history}
    Human: {human_input}
    Assistant:"""

    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2),
    )

    return chatgpt_chain.predict(human_input=input_entry.get())

root = tk.Tk()
root.title("BabyAGI")

# Create an entry field for input
input_entry = tk.Entry(root, width=50)
input_entry.pack()

def button_click():
    response = chatgpt_clone()
    response_label.config(text=response)

# Create a label to display the response
response_label = tk.Label(root, text="", wraplength=400)
response_label.pack()

# Create a button to trigger the backend function
button = tk.Button(root, text="Run", command=button_click)
button.pack()

# Run the Tkinter event loop
root.mainloop()
