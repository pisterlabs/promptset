import streamlit as st
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pdf_processor import get_index_for_pdf
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage, SystemMessage
import os


# title of pqge
st.title("RAG enhanced Gemini Assistant")

# get th emodels we need
load_dotenv(find_dotenv())
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0,google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Cached function to create a vectordb for the provided PDF files
@st.cache_data
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, embeddings
        )
    return vectordb


# Upload PDF files using Streamlit's file uploader
pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

# If PDF files are uploaded, create the vectordb and store it in the session state so that we wouldn't do it each time the app is laoded
if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)




# Get the current prompt from the session state or set a default value
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# Display previous chat messages
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question 
question = st.chat_input("Ask here")

# Handle the user's question
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()

    # Search the vectordb for similar content to the user's question
    search_results = vectordb.similarity_search(question, k=3)
    print(search_results)
    print("----------------------------------------")
    # sources of result
    pdf_extract = "/n ".join([result.page_content for result in search_results])
    print(pdf_extract)
    # Define the template for the chatbot prompt
    prompt_template = f"""
        You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

        Keep your answer short and to the point.
        
        The evidence are the context of the pdf extract with metadata. 
        
        Carefully focus on the metadata specially 'filename' and 'page' whenever answering.
        
        Make sure to add filename and page number at the end of sentence you are citing to.
            
        Reply "Not applicable" if text is irrelevant.

        If youfail to get the information, clearly state so and do not add anything else.
        
        The PDF content is:
        {pdf_extract}
    """

    # # Update the prompt with the pdf extract
    # prompt[0] = {
    #     "role": "system",
    #     "content": prompt_template.format(pdf_extract=pdf_extract),
    # }

    # # Add the user's question to the prompt and display it
    # prompt.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    # Display an empty assistant message while waiting for the response
    with st.chat_message("assistant"):
        botmsg = st.empty()

    response = []
    result = ""
    
    result = model(
        [
            SystemMessage(content=prompt_template.format(pdf_extract=pdf_extract)),
            HumanMessage(content=question),
        ]
    )
    print(result)
    botmsg.write(result.content)

    # for chunk in st.session_state.chat.send_message(prompt):
    #     text = chunk.choices[0].get("delta", {}).get("content")
    #     if text is not None:
    #         response.append(text)
    #         result = "".join(response).strip()
    #         botmsg.write(result)

    # #Add the assistant's response to the prompt
    # prompt.append({"role": "assistant", "content": result})

    # # Store the updated prompt in the session state
    # st.session_state["prompt"] = prompt
    # prompt.append({"role": "assistant", "content": result})

    # # Store the updated prompt in the session state
    # st.session_state["prompt"] = prompt