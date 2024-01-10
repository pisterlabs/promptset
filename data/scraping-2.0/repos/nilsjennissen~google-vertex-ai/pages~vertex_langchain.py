'''
This is the file for the Vertex AI Agent
'''

#%% ---------------------------------------------  IMPORTS  ----------------------------------------------------------#
import streamlit as st
from credentials import OPENAI_API_KEY, project_id
from vertexai.preview.language_models import TextGenerationModel
import tempfile
import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from main import rec_streamlit, speak_answer, get_transcript_whisper
from PyPDF2 import PdfReader

from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA

import os
import time
import vertexai
from vertexai.preview.language_models import ChatModel
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS


#%% -----------------------------------------  GOOGLE VERTEX AI ------------------------------------------------------#
embeddings = VertexAIEmbeddings()

# Initialise the vertexai environment
vertexai.init(project=project_id, location="us-central1")

# Initialise the chat model
model = ChatModel.from_pretrained("chat-bison@001")
chat = model.start_chat(examples=[])

#%% ---------------------------------------------  INTERFACE  --------------------------------------------------------#

# --------------------  SETTINGS  -------------------- #
st.set_page_config(page_title="Home", layout="wide")
st.markdown("""<style>.reportview-container .main .block-container {max-width: 95%;}</style>""", unsafe_allow_html=True)

# --------------------- HOME PAGE -------------------- #
st.title("GOOGLE VERTEX AI LANGCHAIN AGENT")
st.write("""This VERTEX AI Agent reads year selected research papers and tells you everything you need to knwo. Scan 
the knowledge of Google Scholar in seconds!""")
st.write("Let's start interacting with Vertex AI")


# ----------------- SIDE BAR SETTINGS ---------------- #
st.sidebar.subheader("Settings:")
tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=False)
ner_enabled = st.sidebar.checkbox("Enable NER in Response", value=False)

# ------------------ FILE UPLOADER ------------------- #
st.sidebar.subheader("File Uploader:")
uploaded_files = st.sidebar.file_uploader("Choose files", type=["csv", "html", "css", "py", "pdf", "ipynb", "md", "txt"],
                                          accept_multiple_files=True)
st.sidebar.metric("Number of files uploaded", len(uploaded_files))
st.sidebar.color_picker("Pick a color for the answer space", "#C14531")

# ------------------- FILE HANDLER ------------------- #
if uploaded_files:

    file_index = st.sidebar.selectbox("Select a file to display", options=[f.name for f in uploaded_files])
    selected_file = uploaded_files[[f.name for f in uploaded_files].index(file_index)]
    file_extension = selected_file.name.split(".")[-1]

    if file_extension in ["pdf"]:
        try:
            # --- Temporary file save ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(selected_file.getvalue())
                temp_file_path = temp_file.name

            # --- Writing PDF content ---
            with st.expander("Document Expander (Press button on the right to fold or unfold)", expanded=True):
                st.subheader("Uploaded Document:")
                with open(temp_file_path, "rb") as f:
                    pdf = PdfReader(f)
                    for page in pdf.pages:
                        text = page.extract_text()
                        st.write(text)


        except Exception as e:
            st.write(f"Error reading {file_extension.upper()} file:", e)


# --------------------- USER INPUT --------------------- #
user_input = st.text_area("")
# If record button is pressed, rec_streamlit records and the output is saved
audio_bytes = rec_streamlit()

# ------------------- TRANSCRIPTION -------------------- #
if audio_bytes or user_input:

    if audio_bytes:
        try:
            with open("audio.wav", "wb") as file:
                file.write(audio_bytes)
        except Exception as e:
            st.write("Error recording audio:", e)
        transcript = get_transcript_whisper("audio.wav")
    else:
        transcript = user_input

    st.write("**Recognized:**")
    st.write(transcript)

    if any(word in transcript for word in ["abort recording"]):
        st.write("... Script stopped by user")
        exit()


    # ----------------------- ANSWER ----------------------- #
    with st.spinner("Fetching answer ..."):
        time.sleep(6)

        llm = VertexAI()

        # Text Splitter
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Use the PDF agent to answer the question
        docs = knowledge_base.similarity_search(transcript)
        # Show the amount of chunks found
        st.write(f"Found {len(docs)} chunks.")

        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=transcript)
        st.write("**AI Response:**", answer)
        speak_answer(answer, tts_enabled)
        st.success("**Interaction finished**")
