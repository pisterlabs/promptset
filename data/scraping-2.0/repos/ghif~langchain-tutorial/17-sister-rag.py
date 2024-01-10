from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from time import process_time

from langchain import hub
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# Load, chunk and index the contents of the pdf.
pdf_path = "/Users/mghifary/Work/Code/AI/data/SSD_SISTER_BKD.pdf"
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # still not sure what is the best chunk size and overlap
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the pdf.
retriever = vectorstore.as_retriever()

# Define LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Setup prompt
# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum and keep the answer as concise as possible.
# Always say "\n\nTerima kasih atas pertanyaannya. Apakah jawaban kami sudah membantu?" at the end of the answer.

# {context}

# Question: {question}

# Helpful Answer:"""

# prompt = PromptTemplate.from_template(template)
# print(f"\nPrompt example messages: {prompt}")

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
Always say "\n\nTerima kasih atas pertanyaannya. Apakah jawaban kami sudah membantu?" at the end of the answer.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
)

# Test prompt-response
queries = [
    # "bagaimana mendaftar ke program kampus merdeka?",
    "jelaskan tentang BKD",
    "bagaimana cara mengisinya di SISTER?"
]


chat_history = []
for query in queries:
# query = "bagaimana mendaftar ke program kampus merdeka?"
    print(f"\nQuery: {query}\n")

    # # Check retrieved docs
    # retrieved_docs = retriever.invoke(query)
    # for i, rdoc in enumerate(retrieved_docs):
    #     print(f"[{i}] {rdoc.page_content[:100]}")
    # print(f"Retriever docs: {len(retrieved_docs)}")

    start_t = process_time()
    ai_msg = rag_chain.invoke(
        {
            "question": query,
            "chat_history": chat_history
        }
    )
    elapsed_t = process_time() - start_t
    print(f"\nResponse ({elapsed_t:.2f}secs): \n")
    print(f"{ai_msg.content}\n")

    chat_history.extend([HumanMessage(content=query), ai_msg])