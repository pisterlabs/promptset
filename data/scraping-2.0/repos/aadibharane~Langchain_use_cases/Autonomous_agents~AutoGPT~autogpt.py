#Set up tools
#We’ll set up an AutoGPT with a search tool, and write-file tool, and a read-file tool

# from langchain.utilities import SerpAPIWrapper
# from langchain.agents import Tool
# from langchain.tools.file_management.write import WriteFileTool
# from langchain.tools.file_management.read import ReadFileTool
# from serpapi import GoogleSearch
# from langchain.experimental import AutoGPT
# from langchain.chat_models import ChatOpenAI

# import os

# os.environ["OPENAI_API_KEY"] ="OPENAI_API_KEY"

# serpapi_key="serpapi_key"
# search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
# tools = [
#     Tool(
#         name = "search",
#         func=search.run,
#         description="useful for when you need to answer questions about current events. You should ask targeted questions"
#     ),
#     WriteFileTool(),
#     ReadFileTool(),
# ]

# #print(tools)

# #Set up memory
# #The memory here is used for the agents intermediate steps

# from langchain.vectorstores import FAISS
# from langchain.docstore import InMemoryDocstore
# from langchain.embeddings import OpenAIEmbeddings

# # Define your embedding model
# def autogpt():
#     embeddings_model = OpenAIEmbeddings()
#     # Initialize the vectorstore as empty
#     import faiss

#     embedding_size = 1536
#     #index = faiss.IndexFlatL2(embedding_size)
#     index = faiss.IndexFlat(embedding_size, faiss.METRIC_L2)

#     vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
#     #index = faiss.IndexFlat(embedding_size, faiss.METRIC_L2)




#     #Setup model and AutoGPT
#     #Initialize everything! We will use ChatOpenAI model
    
#     agent = AutoGPT.from_llm_and_tools(
#         ai_name="Tom",
#         ai_role="Assistant",
#         tools=tools,
#         llm=ChatOpenAI(temperature=0),
#         memory=vectorstore.as_retriever()
#     )
#     # Set verbose to be true
#     agent.chain.verbose = True

#     #Run an example
#     #Here we will make it write a weather report for SF

#     res=agent.run(input("Enter a a prompt: ")  )#["write a weather report for SF today"])

#     print(res)
#     return res

# autogpt()

import tkinter as tk
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from serpapi import GoogleSearch
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI
import os
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

def run_autogpt(prompt):
    os.environ["OPENAI_API_KEY"] = "sOPENAI_API_KEY"
    serpapi_key = "serpapi_key"
    search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]

    def autogpt():
        embeddings_model = OpenAIEmbeddings()
        import faiss
        embedding_size = 1536
        index = faiss.IndexFlat(embedding_size, faiss.METRIC_L2)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

        agent = AutoGPT.from_llm_and_tools(
            ai_name="Tom",
            ai_role="Assistant",
            tools=tools,
            llm=ChatOpenAI(temperature=0),
            memory=vectorstore.as_retriever()
        )
        agent.chain.verbose = True

        res = agent.run(prompt)
        return res

    return autogpt()

def run_autogpt_gui():
    def submit_prompt():
        prompt = prompt_entry.get()
        output = run_autogpt(prompt)
        output_text.config(state=tk.NORMAL)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, output)
        output_text.config(state=tk.DISABLED)

    # Create the GUI
    window = tk.Tk()
    window.title("AutoGPT with Tkinter")

    prompt_label = tk.Label(window, text="Enter a prompt:")
    prompt_label.pack()

    prompt_entry = tk.Entry(window, width=50)
    prompt_entry.pack()

    submit_button = tk.Button(window, text="Submit", command=submit_prompt)
    submit_button.pack()

    output_label = tk.Label(window, text="Output:")
    output_label.pack()

    output_text = tk.Text(window, width=50, height=10)
    output_text.config(state=tk.DISABLED)
    output_text.pack()

    window.mainloop()

run_autogpt_gui()
