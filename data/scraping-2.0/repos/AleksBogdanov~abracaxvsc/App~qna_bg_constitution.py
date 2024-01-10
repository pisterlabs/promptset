import re

import streamlit as st
from dotenv import load_dotenv
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import openai
from langchain.embeddings.openai import OpenAIEmbeddings  # Wrapper of Embeddings in OpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_chat import message

from utils import pdf_to_string

load_dotenv()  # Loading the API keys safely

st.set_page_config(page_title="Конституция")
st.markdown(
    "<h1 style='text-align: center; color: black;'>Задайте въпрос свързан с Конституцията на България 📜 </h1>",
    unsafe_allow_html=True)
# if "messages" in st.session_state:
#     st.cache_resource.clear()
hide_button_style = """
<style>
  .css-14xtw13 {    
    display: none;
  }
</style>

"""
st.markdown(hide_button_style, unsafe_allow_html=True)

if "msgs" not in st.session_state:
    st.session_state.msgs = []

# Uploading the BG Constitution pdf
pdf_file = "bg_constitution.pdf"
# Reading the file
text = pdf_to_string(pdf_file)

# The Constitution is split by its articles. Because they are separate non-independent entities,
# hey should be treated as a single units. Not to lose their meaning, they oughth not to be
# separated.
sections = re.split(r'(?<=\.)\s+(?=Чл\.\s\d+)', text)

# If the section is however bigger than 1500 tokens, it should be divided further,
# the free-trial version of ChatGPT cannot allow more than 4096 tokens.
# Checking whether chunk is bigger than 1500
bigger_sections = [section for section in sections if len(section) >= 1500]
smaller_sections = [section for section in sections if len(section) < 1500]

# Initialize how to split the these section, using Langchain
# Chunk overlap means, implies that if is set to 100, the chunk will contain
# 50 tokens from the previous and 50 from the next section. The idea is not
# to lose context.
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    chunk_size=1500,
    chunk_overlap=300,
    length_function=len
)

for big_section in bigger_sections:
    chunks = text_splitter.split_text(big_section)
    smaller_sections.extend(chunks)
# Initialize Embeddings -> Converting the text into numbers
embeddings = OpenAIEmbeddings()
# Creating knowledge base. Using facebook FAISS Algorithm. More info in the docs.
knowledge_base = FAISS.from_texts(smaller_sections, embeddings)
message("Как бих могъл да съдействам?", is_user=False)


with st.sidebar:

    # Get the user question
    user_input = st.text_input("Задай своя въпрос: ")
    # Prompt template is created to reduce hallucination, however it does not work well in bulgarian language
    prompt_template = f"""Въпрос: {user_input} отговаряй вежливо и учтиво. Ако не знаеш отговора,
    кажи че не знаеш, не си измисляй. Отговаряй само български."""
    if user_input:
        st.session_state.msgs.append(user_input)
        # Search for similarity in the knowledge base
        try:
            docs = knowledge_base.similarity_search(user_input)
            llm = OpenAI(model_name='text-davinci-003', temperature=0.3)
            chain = load_qa_chain(llm, chain_type="stuff")
            with st.spinner('Мисля ...'):
                response = chain.run(input_documents=docs, question=prompt_template)
            st.session_state.msgs.append(response)
        except:
            pass
# Get messages from session
msgs = st.session_state.get('msgs', [])
# Display messages
for i, msg in enumerate(msgs[0:]):
    if i % 2 == 0:
        message(msg, is_user=True, key=str(i) + '_user')
    else:
        message(msg, is_user=False, key=str(i) + '_ai')


