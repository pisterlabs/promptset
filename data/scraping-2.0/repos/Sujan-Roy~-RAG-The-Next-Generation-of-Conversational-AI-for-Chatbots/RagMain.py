import streamlit as st
import os
import databutton as db
import openai
from RagChatbot import get_index_for_pdf
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI


load_dotenv()

# set the title and description for the app
st.title("PDF Chatbot: interact with your PDFs")
st.markdown("""
1. upload your pdf
2. Ask your questions about the pdf            
3. Hit the submit button and get your answers """)

#set up the OPeN AI API key
#os.environ["OPENAI_API_KEY"] = db.secrets.get("OPENAI_API_KEY")
#openai.api_key = db.secrets.get("OPENAI_API_KEY")
openai.api_key= os.getenv("OPENAI_API_KEY")

# upload the pdf
pdf_files= st.file_uploader("Upload your pdf", type=(['pdf']),accept_multiple_files=True)

# cached function to create a vectorDB for the pdf
@st.cache_data
def create_vector_db(files,filenames):
    """
    Create a vector database for the uploaded PDF files.

    Parameters:
    files (List[File]): List of uploaded PDF files.
    filenames (List[str]): List of filenames corresponding to the uploaded PDF files.

    Returns:
    vector_db (VectorDB): The created vector database for the PDF files.
    """
    # show a spinner while the vectorDB is being created
    with st.spinner("Vector database"):
        # create the vectorDB
        vector_db = get_index_for_pdf(
            [file.getvalue() for file in files], filenames,openai.api_key
            )
    return vector_db

# if pdf is uploaded, create a vectorDB and store it in the session state
if pdf_files:
    pdf_file_names= [file.name for file in pdf_files]
    st.session_state["vector_db"] = create_vector_db(pdf_files,pdf_file_names)


# define the template for the chatbot prompt

prompt_template = """
You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

    Keep your answer short and to the point.
    
    The evidence are the context of the pdf extract with metadata. 
    
    Carefully focus on the metadata specially 'filename' and 'page' whenever answering.
    
    Make sure to add filename and page number at the end of sentence you are citing to.
        
    Reply "Not applicable" if text is irrelevant.
     
    The PDF content is:
    {pdf_extract}
"""
# Get the current prompt from the session state or set a default value
prompt= st.session_state.get("prompt",[{"role":"system","content":"none"}])

#Display previous chat messages
for message in prompt:
    if message["role"]=="system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# get the user question
question = st.chat_input("Ask your question")

# Handle the user input

if question:
    vector_db = st.session_state.get("vector_db")
    if not vector_db:
        with st.warning("Assistant"):
            st.write("Please upload a PDF file first")
            st.stop()

    # Search the vectorDB for the most similar content to the user question
    search_results = vector_db.similarity_search(question, k=3)

    pdf_extract = "\n".join([result.page_content for result in search_results])

    # Update the prompt with the pdf extract
    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }

    # LLM part
    # Note that the ChatCompletion is used as it was found to be more effective to produce good results.
    # Using just Completion often resulted in exceeding token limits.
    # According to https://platform.openai.com/docs/models/gpt-3-5

    # Add the user question to the prompt and display it
    prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Display an empty assistant message while waiting for the response from the API
    with st.chat_message("Assistant"):
        st.write("Thinking...")
        bot_msg = st.empty()

    # Call ChatGPT with streaming and display the response as it comes in
    response = []
    result = ""
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=prompt,
        stream=True,
    ):
        text = chunk.choices[0].get("message", {}).get("content")
        if text is not None:
            response.append(text)
            result = "".join(response).strip()
            bot_msg.write(result)

    # Add the assistant's response to the prompt
    prompt.append({"role": "Assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt

