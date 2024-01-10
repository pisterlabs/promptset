import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
import google.generativeai as genai
import streamlit as st
genai.configure(api_key=st.secrets["google_api_key"])


# Prompt and Memory
# Template prompt
template = """Jawablah pertanyaan di bawah ini berdasarkan konteks yang diberikan! \
Jika dalam pertanyaan merujuk ke histori chat sebelumnya, maka gunakan konteks dari pertanyaan \
sebelumnya untuk menjawab! 
Jika konteks tidak mengandung jawaban yang diinginkan, maka buatlah jokes saja dengan pertanyaan \
tersebut agar suasana jadi cair!
Konteks:
{context}

Pertanyaan:
{question}
"""

template_system = """Namamu adalah FTMMQA, sebuah chatbot Fakultas Teknologi Maju dan \
Multidisiplin (FTMM), Universitas Airlangga. Kamu siap menjawab pertanyaan apapun \
seputar FTMM. Kamu menjawab setiap pertanyaan dengan ceria, sopan, dan asik!

Ketika kamu menjawab pertanyaan, jangan pernah bilang bahwa jawabanmu itu didasarkan \
pada konteks yang diberikan, berlagaklah seolah kamu memang mengerti segalanya \
tentang FTMM. Tetapi jika informasi tidak disediakan pada konteks yang diberikan \
jawablah dengan lucu saja! mungkin kamu bisa membuat jokes dari pertanyaan user \ 
atau pura-pura lupa.
"""

# Prompt
prompt_template = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(template_system),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(template),
    ]
)

# Memory
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=2)


# Embedding function
def embed_fn(text):
    embeddings = genai.embed_content(model="models/embedding-001",
        content=text,
        task_type="retrieval_document")["embedding"]
    return embeddings