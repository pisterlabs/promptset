import streamlit as st

hide_streamlit_style = """
            <style>
            #MainMenu  {visibility: hidden;}
            footer  {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def free_version():  

        import streamlit as st
        import pandas as pd
        from pdf2image import convert_from_path
        import pytesseract
        from PyPDF2 import PdfReader
        from langchain.document_loaders import PyPDFLoader
        from langchain.agents import create_csv_agent
        from langchain.llms import OpenAI
        import os
        from apikey import apikey
        from langchain.document_loaders import TextLoader
        from langchain.indexes import VectorstoreIndexCreator
        from langchain.text_splitter import CharacterTextSplitter
        import time
        from langchain import HuggingFaceHub
        from langchain.embeddings import HuggingFaceInstructEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains import RetrievalQA
        import textwrap
        import os 
        from langchain.document_loaders import DirectoryLoader
        import shutil
        from deep_translator import GoogleTranslator 
        from langdetect import detect
        HUGGINGFACE_API_TOKEN = "your Huggingfaceaccesstoken" #you can get it from huggingface for free
        repo_id = "tiiuae/falcon-7b-instruct"
        st.title("MKG: Your Research Chat Buddy 📄🤖")
        llm=HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                                repo_id=repo_id,
                                model_kwargs={"temperature":0.5, "max_new_tokens":1000})
        
        
#os.environ['OPENAI_API_KEY'] = apikey
        pdfs_directory = "PDFs"
        if not os.path.exists(pdfs_directory):
            os.makedirs(pdfs_directory)
        for file_name in os.listdir(pdfs_directory):
                                file_path = os.path.join(pdfs_directory, file_name)
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
        #Free_Open Source Model
        
        if 'exit' not in st.session_state:
                st.session_state['exit'] = False
        def typewriter(text: str, speed: float):
            container = st.empty()
            displayed_text = ""

            for char in text:
                displayed_text += char
                container.markdown(displayed_text)
                time.sleep(1/speed)
        def wrap_text_preserve_newlines(text, width=110):
            # Split the input text into lines based on newline characters
            lines = text.split('\n')

            # Wrap each line individually
            wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

            # Join the wrapped lines back together using newline characters
            wrapped_text = '\n'.join(wrapped_lines)

            return wrapped_text            
                # do something with the data
        def process_llm_response(llm_response,llm_originalresponse2):
            result_text = wrap_text_preserve_newlines(llm_originalresponse2)
            typewriter(result_text, speed=40)
            st.write('\n\nSources:')
            for source in llm_response["source_documents"]:
                typewriter(source.metadata['source'], speed=35)
            
            
        def save_uploaded_pdfs(uploaded_files):
        # Save uploaded PDF files to the "PDFs" directory
            if uploaded_files:
                    for uploaded_file in uploaded_files:
                        original_filename = uploaded_file.name  # Get the original filename
                        unique_filename = original_filename
                        pdf_path = os.path.join(pdfs_directory, unique_filename)
                        
                        # Extract the content from the UploadedFile
                        file_content = uploaded_file.read()
                        
                        with open(pdf_path, "wb") as pdf_file:
                            pdf_file.write(file_content)
                        success_message = st.empty()
                        success_message.success(f"File '{unique_filename}' successfully uploaded.")
                        time.sleep(10)  # Adjust the duration as needed
                        success_message.empty()
        def launch(): 
            
            uploaded_files = st.file_uploader("Please upload all your documents at once", type=["pdf"], accept_multiple_files=True)
            original_question = st.text_input("Once uploaded, you can chat with your document. Enter your question here or type exit to end and upload new documents:")
            question = GoogleTranslator(source='auto', target='en').translate(original_question)
            submit_button = st.button('Submit')
            if uploaded_files and submit_button:
                                save_uploaded_pdfs(uploaded_files)
                                loader = DirectoryLoader('./PDFs', glob="./*.pdf", loader_cls=PyPDFLoader)

                                documents = loader.load()
                                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                                texts = text_splitter.split_documents(documents)
                                instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                                                    model_kwargs={"device": "cuda"})
                                persist_directory = 'db'

                                ## Here is the new embeddings being used
                                embedding = instructor_embeddings

                                vectordb = Chroma.from_documents(documents=texts,
                                                                embedding=embedding)

                                retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                                
                                qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                                                    chain_type="stuff",
                                                                    retriever=retriever,
                                                                    return_source_documents=True)
                                # Initial state
                                
                                while st.session_state['exit'] == False:
                                    
                                            #question = st.text_input("Once uploaded, you can chat with your document. Enter your question here or type exit to end and upload new documents:", key=f"question_input_{i}")
                                            with st.spinner('Generating Answer...'):
                                            
                                                if question.lower() == 'exit':
                                                    st.session_state['exit'] = True
                                                
                                                else:
                                                    detected_source_language = detect(original_question)
                                                    chunk_size = 5000
                                                    # Process the question and display the response
                                                    llm_originalresponse = qa_chain(question)
                                                    llm_originalresponse2=str(llm_originalresponse['result'])

                                                    chunks = [llm_originalresponse2[i:i+chunk_size] for i in range(0, len(llm_originalresponse2), chunk_size)]
                                                    translated_chunks = []
                                                    my_translator=GoogleTranslator(source='auto', target=detected_source_language)
                                                    for chunk in chunks:
                                                            translated_chunk = my_translator.translate(chunk)
                                                            translated_chunks.append(translated_chunk)
                                                    llm_originalresponse2=''.join(translated_chunks)
                                                    process_llm_response(llm_originalresponse,llm_originalresponse2)
                                                    print(repo_id)
                                                    break
                                        
                                                if st.session_state['exit'] == True:
                                                    output="Thank you for trying our Tool, We hope you liked it"
                                                    typewriter(output, speed=5)
                                                    # Delete files and folders
                                                    for file_name in os.listdir(pdfs_directory):
                                                        file_path = os.path.join(pdfs_directory, file_name)
                                                        if os.path.isfile(file_path):
                                                            os.remove(file_path)
                                                    
                                                    # Remove "db" directory
                                                
                                                    
                                                    break
                                        
                                    
                
            st.warning("⚠️ Please Keep in mind that the accuracy of the response relies on the :red[PDF's Quality] and the :red[prompt's Quality]. Occasionally, the response may not be entirely accurate. Consider using the response as a reference rather than a definitive answer.")  
                            
                            
        launch()   
           
def paid_version():
        import streamlit as st
        import pandas as pd
        from pdf2image import convert_from_path
        import pytesseract
        from PyPDF2 import PdfReader
        from langchain.agents import create_csv_agent
        from langchain.llms import OpenAI
        import os
        from apikey import apikey
        from langchain.document_loaders import TextLoader
        from langchain.indexes import VectorstoreIndexCreator
        import time
        from langchain import HuggingFaceHub
        from PyPDF2 import PdfReader
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.vectorstores import FAISS
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI
        import streamlit as st
        def set_openAi_api_key(api_key: str):
            st.session_state["OPENAI_API_KEY"] = api_key
            os.environ['OPENAI_API_KEY'] = api_key
        def openai_api_insert_component():
            with st.sidebar:
                st.markdown(
                    """
                    ## Quick Guide 🚀
                    1. Get started by adding your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑
                    2. Easily upload a PDF document📄
                    3. Engage with the content - ask questions, seek answers💬
                    """
                )

                api_key_input = st.text_input("Input your OpenAI API Key",
                                            type="password",
                                            placeholder="Format: sk-...",
                                            help="You can get your API key from https://platform.openai.com/account/api-keys.")
                
                
                if api_key_input == "" or api_key_input is None:
                        st.sidebar.caption("👆 :red[Please set your OpenAI API Key here]")
                
                
                st.caption(":green[Your API is not stored anywhere. It is only used to generate answers to your questions.]")

                set_openAi_api_key(api_key_input)
        def get_response_from_OpenAI_LangChain(uploaded_file, prompt):
    
    
            reader = PdfReader(uploaded_file)

            raw_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text

            text_splitter = CharacterTextSplitter(separator = "\n",
                                                chunk_size = 1000,
                                                chunk_overlap = 200,
                                                length_function = len)

            texts = text_splitter.split_text(raw_text)
            with st.spinner('Processing Embeddings...'):
                embeddings = OpenAIEmbeddings()
                doc_search = FAISS.from_texts(texts, embeddings)
                chain = load_qa_chain(OpenAI(), chain_type='map_reduce')

            query = prompt
            docs = doc_search.similarity_search(query)

            with st.spinner('Generating Answer...'):
                response = chain.run(input_documents=docs, question=query) # response
                

                
                
                st.session_state['response'] = response
            
                return response 
            
        def load_csv_data(uploaded_file):
            df = pd.read_csv(uploaded_file)
            df.to_csv("uploaded_file.csv")
            return df
        
        def launchpaidapp():
            st.title("MKG: Your Research Chat Buddy 📄🤖")
            openai_api_insert_component()
            os.environ['OPENAI_API_KEY'] = st.session_state['OPENAI_API_KEY']
            uploaded_file = st.file_uploader("Upload a CSV or PDF file", type=["csv", "pdf"], accept_multiple_files=False)
            prompt = st.text_input("Enter your question here:")
            submit_button = st.button('Submit')
            if submit_button:
                if uploaded_file is not None:
                    if uploaded_file.type == "text/csv":
                        #doc = "csv"
                        data = load_csv_data(uploaded_file)
                        agent = create_csv_agent(OpenAI(temperature=0), 'uploaded_file.csv', verbose=True)
                        st.dataframe(data)
                        response = agent.run(prompt)
                elif uploaded_file.type == "application/pdf":
                    response = get_response_from_OpenAI_LangChain(uploaded_file, prompt)
                    if response:
                        st.write(response)           
            st.warning("⚠️ Please Keep in mind that the accuracy of the response relies on the :red[PDF's Quality] and the :red[prompt's Quality]. Occasionally, the response may not be entirely accurate. Consider using the response as a reference rather than a definitive answer.")       
        
        launchpaidapp()
def intro():
            st.markdown("""
            # MKG: Your Research Chat Buddy 📄🤖

            Welcome to MKG-Assistant, where AI meets your Documents! 🚀🔍

            ## Base Models

            Q&A-Assistant is built on OpenAI's GPT 3.5 for the premium version and Falcon 7B instruct Model for the free version to enhance your research experience. Whether you're a student, researcher, or professional, we're here to simplify your interactions with your documents. 💡📚

            ## Standout Features

            - AI-Powered Q&A: Upload your PDF (supports also CSV in the premium version), enter your API key, and ask questions. Get precise answers like a personal Q&A expert! 💭🤖

            ## How to Get Started

            1. Upload your Document.
            2. Enter your API key.(Only if you chose the premium version. Key is not needed in the free version)
            3. Ask questions using everyday language.
            4. Get detailed, AI-generated answers.

            5. Enjoy a smarter way to read PDFs!

            ## Explore More

            - Open Source Edition: Free with basic features.
            - Premium Edition: Unlocks advanced capabilities, including CSV and spreadsheet Q&A, using your OpenAI API key.

            ## It is Time to Dive in!


            """)                
page_names_to_funcs = {
    "Main Page": intro,
    "Open Source Edition (Free version)": free_version,
    "Premium edition (Requires Open AI API Key )": paid_version
    
}


    
    
    
    

demo_name = st.sidebar.selectbox("Choose a version", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()    
st.sidebar.markdown('<a href="https://www.linkedin.com/in/mohammed-khalil-ghali-11305119b/"> Connect on LinkedIn <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="30" height="30"></a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="https://github.com/khalil-ghali"> Check out my GitHub <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" width="30" height="30"></a>', unsafe_allow_html=True)
