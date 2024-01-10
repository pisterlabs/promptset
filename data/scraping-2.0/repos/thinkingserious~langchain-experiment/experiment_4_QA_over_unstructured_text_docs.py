import logging
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()

# https://integrations.langchain.com/
# https://python.langchain.com/docs/modules/data_connection/document_loaders
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query("What is Task Decomposition?"))
# >> Task decomposition is a technique used to break down complex tasks into smaller and simpler steps. 
# >> It can be done using LLM with simple prompting, task-specific instructions, or with human input.

data = loader.load()

# https://python.langchain.com/docs/modules/data_connection/document_transformers
# https://python.langchain.com/docs/use_cases/question_answering/how_to/document-context-aware-QA
# https://python.langchain.com/docs/use_cases/question_answering/docs/integrations
# https://python.langchain.com/docs/integrations/document_loaders/grobid
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

# https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html
# https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.chroma.Chroma.html
# https://integrations.langchain.com/vectorstores
# https://python.langchain.com/docs/modules/data_connection/vectorstores
# https://integrations.langchain.com/embeddings
# https://python.langchain.com/docs/modules/data_connection/text_embedding
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# https://www.pinecone.io/learn/what-is-similarity-search
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
print(docs)

# >> [Document(page_content='Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.', metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"}), Document(page_content='Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.', metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"}), Document(page_content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.', metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"}), Document(page_content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#', metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"})]

print(len(docs))
# >> 4 

# https://twitter.com/karpathy/status/1647025230546886658?s=20
# https://python.langchain.com/docs/modules/data_connection/retrievers
# https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.svm.SVMRetriever.html
# https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
# https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
# https://python.langchain.com/docs/use_cases/question_answering/how_to/document-context-aware-QA
svm_retriever = SVMRetriever.from_documents(all_splits,OpenAIEmbeddings())
docs_svm=svm_retriever.get_relevant_documents(question)
print(docs_svm)
# >> [Document(page_content='Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.', metadata={}), Document(page_content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#', metadata={}), Document(page_content='Challenges in long-term planning and task decomposition: Planning over a lengthy history and effectively exploring the solution space remain challenging. LLMs struggle to adjust plans when faced with unexpected errors, making them less robust compared to humans who learn from trial and error.', metadata={}), Document(page_content='Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.', metadata={})]
print(len(docs_svm))
# >> 4 

# https://api.python.langchain.com/en/latest/chat_models/langchain.chat_models.openai.ChatOpenAI.html
# https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.multi_query.MultiQueryRetriever.html
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),
                                                  llm=ChatOpenAI(temperature=0))
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
print(unique_docs)
# >> INFO:langchain.retrievers.multi_query:Generated queries: ['1. How can Task Decomposition be approached?', '2. What are the different methods for Task Decomposition?', '3. What are the various approaches to decomposing tasks?']
# >> [Document(page_content='Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.', metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"}), Document(page_content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.', metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"}), Document(page_content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#', metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"}), Document(page_content='Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.', metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"})]
print(len(unique_docs))
# >> 4 

# https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html
# https://api.python.langchain.com/en/latest/chat_models/langchain.chat_models.openai.ChatOpenAI.html
# https://integrations.langchain.com/llms
# https://integrations.langchain.com/chat-models
# https://python.langchain.com/docs/modules/model_io/models
# https://github.com/imartinez/privateGPT
# https://github.com/nomic-ai/gpt4all
# (left off here) https://python.langchain.com/docs/use_cases/question_answering/#go-deeper-4
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
print(qa_chain({"query": question}))
# >> {'query': 'What are the approaches to Task Decomposition?', 'result': 'The approaches to task decomposition include:\n\n1. Prompting with LLM: This approach involves using simple prompts to guide the model in decomposing the task into subgoals or steps. For example, the model can be prompted with instructions like "Steps for XYZ" or "What are the subgoals for achieving XYZ?"\n\n2. Task-specific instructions: In this approach, task-specific instructions are provided to the model to guide the task decomposition process. For example, if the task is to write a novel, the model can be instructed to "Write a story outline" as a step in the task decomposition.\n\n3. Human inputs: Task decomposition can also be done with the help of human inputs. Humans can provide guidance and input to the model in breaking down the task into smaller and simpler steps.\n\nIt\'s important to note that these approaches are not mutually exclusive and can be used in combination depending on the specific requirements of the task and the capabilities of the model.'}

# https://huggingface.co/TheBloke/Nous-Hermes-13B-GGML/blob/main/nous-hermes-13b.ggmlv3.q4_0.bin
# https://api.python.langchain.com/en/latest/llms/langchain.llms.gpt4all.GPT4All.html
# https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html
llm = GPT4All(model="./nous-hermes-13b.ggmlv3.q4_0.bin",max_tokens=2048)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
print(qa_chain({"query": question}))
# >> {'query': 'What are the approaches to Task Decomposition?', 'result': 'There are three main approaches to Task Decomposition: (1) using simple prompts like "Steps for XYZ.\\n1.", (2) with task-specific instructions, such as "Write a story outline." for writing a novel or by human inputs. LLM can also be used for this purpose and it involves breaking down the problem into multiple thought steps and generating multiple thoughts per step to create a tree structure using techniques like CoT or Tree of Thoughts (Yao et al. 2023).'}

# https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html
# https://api.python.langchain.com/en/latest/prompts/langchain.prompts.prompt.PromptTemplate.html
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
result = qa_chain({"query": question})
print(result["result"])
# >> The approaches to Task Decomposition include using simple prompting with LLM, task-specific instructions, and human inputs. Thanks for asking!

# https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever(),
                                       return_source_documents=True)
result = qa_chain({"query": question})
print(len(result['source_documents']))
print(result['source_documents'][0])
# >> 4
# >> page_content='Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.' metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"}

# https://api.python.langchain.com/en/latest/chains/langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain.html
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=vectorstore.as_retriever())
result = qa_chain({"question": question})
print(result)
# >> {'question': 'What are the approaches to Task Decomposition?', 'answer': 'The approaches to Task Decomposition include:\n1. Using LLM with simple prompting, such as providing steps or subgoals for achieving a task.\n2. Using task-specific instructions, such as providing a specific instruction like "Write a story outline" for writing a novel.\n3. Using human inputs to decompose the task.\nAnother approach is the Tree of Thoughts, which extends the Chain of Thought (CoT) technique by exploring multiple reasoning possibilities at each step and generating multiple thoughts per step, creating a tree structure. The search process can be BFS or DFS, and each state can be evaluated by a classifier or majority vote.\nSources: https://lilianweng.github.io/posts/2023-06-23-agent/', 'sources': ''}

# https://python.langchain.com/docs/use_cases/question_answering/how_to/question_answering.html
chain = load_qa_chain(llm, chain_type="stuff")
result = chain({"input_documents": unique_docs, "question": question}, return_only_outputs=True)
print(result)
# >> {'output_text': 'The approaches to task decomposition mentioned in the provided context are:\n\n1. Chain of thought (CoT): This approach involves instructing the language model to "think step by step" and decompose complex tasks into smaller and simpler steps. It enhances model performance on complex tasks by utilizing more test-time computation.\n\n2. Tree of Thoughts: This approach extends CoT by exploring multiple reasoning possibilities at each step. It decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS or DFS, and each state is evaluated by a classifier or majority vote.\n\n3. LLM with simple prompting: This approach involves using simple prompts like "Steps for XYZ" or "What are the subgoals for achieving XYZ" to guide the language model in task decomposition.\n\n4. Task-specific instructions: This approach involves providing task-specific instructions to guide the language model in task decomposition. For example, providing the instruction "Write a story outline" for the task of writing a novel.\n\n5. Human inputs: Task decomposition can also be done with human inputs, where humans provide guidance and instructions to break down a task into smaller subtasks.'}

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectorstore.as_retriever(),
                                       chain_type="stuff")
result = qa_chain({"query": question})
print(result)
# >> {'query': 'What are the approaches to Task Decomposition?', 'result': 'The approaches to task decomposition include:\n\n1. Prompting with LLM: This approach involves using simple prompts to guide the model in decomposing the task into subgoals or steps. For example, the model can be prompted with instructions like "Steps for XYZ" or "What are the subgoals for achieving XYZ?"\n\n2. Task-specific instructions: In this approach, task-specific instructions are provided to the model to guide the task decomposition process. For example, if the task is to write a novel, the model can be instructed to "Write a story outline" as a step in the task decomposition.\n\n3. Human inputs: Task decomposition can also be done with the help of human inputs. Humans can provide guidance and input to the model in breaking down the task into smaller and simpler steps.\n\nIt\'s important to note that these approaches are not mutually exclusive and can be used in combination depending on the specific requirements of the task and the capabilities of the model.'}
