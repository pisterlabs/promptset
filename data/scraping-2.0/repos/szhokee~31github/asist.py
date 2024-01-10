import requests
from bs4 import BeautifulSoup

from langchain.document_loaders import YoutubeLoader
from langchain.schema import Document

from youtuber.api import fetch_youtube_captions
from youtuber import fetch_youtube_captions



from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from sklearn.cluster import KMeans
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import numpy as np
from langchain.chains import ConversationalRetrievalChain
import logging
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

logging.basicConfig(level=logging.INFO) 
# Initialize global variables
global_chromadb = None
global_documents = None
global_short_documents = None

video_url = "https://youtu.be/j5wVGQQ5e5A?si=3D_O4833OxvO0fRn"
captions = fetch_youtube_captions(video_url)
print(captions)

def get_youtube_video_title(video_url):
    response = requests.get(video_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('meta', property='og:title')
    return title['content'] if title else "Title not found"

def fetch_youtube_captions(video_url):
    title = get_youtube_video_title(video_url)
    loader = YoutubeLoader.from_youtube_url(video_url)
    docs = loader.load()
    if docs and len(docs) > 0:
        intro_sentence = "This is the title of the video/transcription/conversation: "
        title_content = intro_sentence + title
        docs[0] = Document(page_content=title_content + "\n\n" + docs[0].page_content)
    return docs

# Initialize the memory outside the function so it persists across different calls
conversation_memory = ConversationBufferMemory(
    memory_key="chat_history",
    max_len=50,
    input_key="question",
    output_key="answer",
    return_messages=True,
)

# Function to reset global variables
def reset_globals():
    global global_chromadb, global_documents, global_short_documents
    global_chromadb = None
    global_documents = None
    global_short_documents = None
    # Reset the conversation memory
    if conversation_memory:
        conversation_memory.clear()

def init_chromadb(openai_api_key):
    global global_chromadb, global_short_documents
    if global_chromadb is None and global_short_documents is not None:
        global_chromadb = Chroma.from_documents(documents=global_short_documents, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))



def process_and_cluster_captions(captions, openai_api_key, num_clusters=12):
    global global_documents, global_short_documents 
    logging.info("Processing and clustering captions")
    
    # Log the first 500 characters of the captions to check their format
    logging.info(f"Captions received (first 500 characters): {captions[0].page_content[:500]}")
    caption_content = captions[0].page_content

    # Ensure captions is a string before processing
    if not isinstance(caption_content, str):
        logging.error("Captions are not in the expected string format")
        return []
    
    # Create longer chunks for summary
    summary_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
    summary_docs = summary_splitter.create_documents([caption_content])

    # Create shorter chunks for QA
    qa_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
    qa_docs = qa_splitter.create_documents([caption_content])
    
    # Process for summary
    summary_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_documents([x.page_content for x in summary_docs])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(summary_embeddings)
    closest_indices = [np.argmin(np.linalg.norm(summary_embeddings - center, axis=1)) for center in kmeans.cluster_centers_]
    representative_docs = [summary_docs[i] for i in closest_indices]

    # Store documents globally
    global_documents = summary_docs  # For summary
    global_short_documents = qa_docs  # For QA

    init_chromadb(openai_api_key)  # Initialize database with longer chunks
    return representative_docs




def generate_summary(representative_docs, openai_api_key, model_name):
    logging.info("Generating summary")
    llm4 = ChatOpenAI(model_name=model_name, temperature=0.2, openai_api_key=openai_api_key)

    # Concatenate texts for summary
    summary_text = "\n".join([doc.page_content for doc in representative_docs])

    summary_prompt_template = PromptTemplate(
        template=(
            "Create a concise summary of a podcast conversation based on the text provided below. The text consists of selected, representative sections from different parts of the conversation. "
            "Your task is to synthesize these sections into a single cohesive and concise summary. Focus on the overarching themes and main points discussed throughout the podcast. "
            "The summary should give a clear and complete understanding of the conversation's key topics and insights, while omitting any extraneous details. It should be engaging and easy to read, ideally in one or two paragraphs. Keep it short where possible"
            "\n\nSelected Podcast Sections:\n{text}\n\nSummary:"
        ),
        input_variables=["text"]
    )
    # Load summarizer chain
    summarize_chain = load_summarize_chain(llm=llm4, chain_type="stuff", prompt=summary_prompt_template)

    # Run the summarizer chain
    summary = summarize_chain.run([Document(page_content=summary_text)])

    logging.info("Summary generation completed")
    return summary





def answer_question(question, openai_api_key, model_name):
    llm4 = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)
    global global_chromadb, global_short_documents

    if global_chromadb is None and global_short_documents is not None:
        init_chromadb(openai_api_key, documents=global_short_documents)
    
    logging.info(f"Answering question: {question}")
    chatTemplate = """
    You are an AI assistant tasked with answering questions based on context from a podcast conversation. Use the provided context and relevant chat messages to answer. If unsure, say so. Keep your answer to three sentences or less, focusing on the most relevant information.
    Chat Messages (if relevant): {chat_history}
    Question: {question} 
    Context from Podcast: {context} 
    Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question", "chat_history"],template=chatTemplate)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm4, 
        chain_type="stuff", 
        retriever=global_chromadb.as_retriever(search_type="mmr", search_kwargs={"k":12}),
        memory=conversation_memory,
        #return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
    )
    # Log the current chat history
    current_chat_history = conversation_memory.load_memory_variables({})
    logging.info(f"Current Chat History: {current_chat_history}")
    response = qa_chain({"question": question}) 
    logging.info(f"this is the result: {response}")
    output = response['answer']    

    return output




import streamlit as st
from youtuber import fetch_youtube_captions
from agent import process_and_cluster_captions, generate_summary, answer_question, reset_globals

# Set Streamlit page configuration with custom tab title
st.set_page_config(page_title="🏄GPTpod", page_icon="🏄", layout="wide")

def user_query(question, openai_api_key, model_name):
    """Process and display the query response."""
    # Add the user's question to the conversation
    st.session_state.conversation.append((f"{question}", "user-message"))

    # Check if this query has been processed before
    if question not in st.session_state.processed_questions:
        # Process the query
        answer = answer_question(question, openai_api_key, model_name)
        if isinstance(answer, str):
            st.session_state.conversation.append((f"{answer}", "grimoire-message"))
        else:
            st.session_state.conversation.append(("Could not find a proper answer.", "grimoire-message"))
        
        st.rerun()

        # Mark this question as processed
        st.session_state.processed_questions.add(question)


# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
    st.session_state.asked_questions = set()
    st.session_state.processed_questions = set()

# Sidebar for input and operations
with st.sidebar:
    st.title("GPT Podcast Surfer🌊🏄🏼")
    st.image("img.png") 

    # Expandable Instructions
    with st.expander("🔍 How to use:", expanded=False):
        st.markdown("""
            - 🔐 **Enter your OpenAI API Key.**
            - 📺 **Paste a YouTube URL.**
            - 🏃‍♂️ **Click 'Run it' to process.**
            - 🕵️‍♂️ **Ask questions in the chat.**
        """)

    # Model selection in the sidebar
    model_choice = st.sidebar.selectbox("Choose Model:", 
                                        ("GPT-4 Turbo", "GPT-3.5 Turbo"), 
                                        index=0)  # Default to GPT-4 Turbo

    # Map friendly names to actual model names
    model_name_mapping = {
        "GPT-4 Turbo": "gpt-4-1106-preview",
        "GPT-3.5 Turbo": "gpt-3.5-turbo"
    }

    selected_model = model_name_mapping[model_choice]
    st.session_state['selected_model'] = model_name_mapping[model_choice]


    # Input for OpenAI API Key
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    # Save the API key in session state if it's entered
    if openai_api_key:
        st.session_state['openai_api_key'] = openai_api_key

    youtube_url = st.text_input("Enter YouTube URL:")

    # Button to trigger processing
    if st.button("🚀Run it"):
        if openai_api_key:
            if youtube_url and 'processed_data' not in st.session_state:
                reset_globals()
                with st.spinner('👩‍🍳 GPT is cooking up your podcast... hang tight for a few secs🍳'):
                    captions = fetch_youtube_captions(youtube_url)
                    if captions:
                        representative_docs = process_and_cluster_captions(captions, st.session_state['openai_api_key'])
                        summary = generate_summary(representative_docs, st.session_state['openai_api_key'], selected_model)
                        st.session_state.processed_data = (representative_docs, summary)
                        if 'summary_displayed' not in st.session_state:
                            st.session_state.conversation.append((f"Here's a rundown of the conversation: {summary}", "summary-message"))
                            guiding_message = "Feel free to ask me anything else about it! :)"
                            st.session_state.conversation.append((guiding_message, "grimoire-message"))
                            st.session_state['summary_displayed'] = True
                    else:
                        st.error("Failed to fetch captions.")
        else:
            st.warning("Please add the OpenAI API key first.")


# Main app logic
for message, css_class in st.session_state.conversation:
    role = "assistant" if css_class in ["grimoire-message", "summary-message", "suggestion-message"] else "user"
    with st.chat_message(role):
        st.markdown(message)


# Chat input field
if prompt := st.chat_input("Ask me anything about the podcast..."):
    user_query(prompt, st.session_state.get('openai_api_key', ''), st.session_state.get('selected_model', 'gpt-4-1106-preview'))
