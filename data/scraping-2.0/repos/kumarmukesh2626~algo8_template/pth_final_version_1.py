import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import VectorStore
from  langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import json
from langchain.prompts import PromptTemplate

import os
import datetime
from langchain.prompts import PromptTemplate
import time  # Import the time module
from langchain.document_loaders import UnstructuredExcelLoader
from io import BytesIO  # Import BytesIO for handling Streamlit UploadedFile

import os
from openpyxl import Workbook
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
openai_api_key = os.getenv("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")
import openai

# Set your OpenAI API key here
openai_api_key = "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"

# Initialize OpenAI API with the key
openai.api_key = openai_api_key

# Now you can use OpenAI API methods


key = '877d123aa6b14f679d5ff6322665e056'
endpoint = 'https://algo8genai.cognitiveservices.azure.com/'

# Import the required variables from html_template

st.set_page_config(page_title='Algo8 Docsgpt', page_icon=':books:')
st.title('Algo8 Docsgpt :books:')
css = '''
<style> 
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b3  13e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px; 
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://w7.pngwing.com/pngs/340/946/png-transparent-avatar-user-computer-icons-software-developer-avatar-child-face-heroes.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://repository-images.githubusercontent.com/156847937/2ac66980-0f3d-11eb-8e62-693087aa1f67">
    </div>
    <div class="message">
        <ul>
            {{MSG}}
        </ul>
    </div>
</div>

'''

# You can now use css, user_template, and bot_template in your project.py code
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

if 'conversation' not in st.session_state:
    st.session_state.conversation = None  # Initialize conversation as None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = None

# Maintain a variable for selected PDFs

#def get_raw_text(uploaded_files, selected_pdfs):
    #text = ''
    #for pdf in uploaded_files:
   #     pdf_name = pdf.name
 #       if pdf_name in selected_pdfs:
    #        pdf_reader = PdfFileReader(pdf)
      #      for page in pdf_reader.pages:
     #           pdf_text += page.extractText()
   # return text



# use your `key` and `endpoint` environment variables



def analyze_local_pdf(uploaded_file, selected_pdfs, output_directory):
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    # Create a new workbook and get the active sheet
    workbook = Workbook()
    sheet = workbook.active

    # Initialize the last row index for the first table
    last_row_index = 1

    for pdf in uploaded_file:
        pdf_name = pdf.name  # Get the filename from the path
        if pdf_name in selected_pdfs:
            pdf_file = BytesIO(pdf.read())
            poller = document_analysis_client.begin_analyze_document(
                    "prebuilt-layout", document=pdf_file
                )
            result = poller.result()

            for table_idx, table in enumerate(result.tables):
                # Add a blank row between tables
                if table_idx > 0:
                    sheet.append([])
                    last_row_index += 1  # Increment the last row index

                for cell in table.cells:
                    # Calculate the adjusted row index based on the difference with the last row index
                    row_index = last_row_index + int(cell.row_index)
                    col_index = int(cell.column_index) + 1  # Add 1 to start from column 1
                    content = cell.content

                    # Fill in the cell at the adjusted row and column indices
                    sheet.cell(row=row_index, column=col_index, value=content)

                # Update the last row index for the next iteration
                last_row_index = row_index

    # Save the workbook to a single Excel file
    path=selected_pdfs[0]
    paths=path[:-4]
    excel_name = paths + ".xlsx"
    print(excel_name)
    excel_path = os.path.join(output_directory, excel_name)
    workbook.save(excel_path)

    print(f"All tables data saved to {excel_path}")
    return excel_path





def text_chunks(raw_text):
    #st.write(raw_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=480, chunk_overlap=200)
    
    chunks = text_splitter.split_text(raw_text)
    docs = text_splitter.create_documents(chunks)

    return docs






def process_selected_pdfs(uploaded_files, selected_pdfs):
    selected_texts = {}
    conversational_summary = ""  # Initialize conversational_summary

    for pdf in uploaded_files:
        pdf_name = pdf.name
        if pdf_name in selected_pdfs:
            pdf_reader = PdfReader(pdf)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

            # Remove extra spaces using regex
            pdf_text_cleaned = re.sub(r'\s+', ' ', pdf_text).strip()
            
            selected_texts[pdf_name] = pdf_text_cleaned

            # Append to conversational_summary
            conversational_summary += f"**Document Name:** {pdf_name}\n"
            conversational_summary += f"{pdf_text_cleaned}\n\n"
            
    return selected_texts, conversational_summary
# Example usage:


from tenacity import retry, retry_if_exception_type, stop_after_attempt


def get_vector_store(docs_1):
    embeddings = OpenAIEmbeddings(openai_api_key='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')

    num_total_characters = sum([len(x.page_content) for x in docs_1])

    docsearch = FAISS.from_documents(docs_1, embeddings)
    return docsearch
    


    
    #print ("Preview:")
   # print (docs[0].page_content, "\n")
 #   print (docs[1].page_content)
    # Get the total number of characters so we can see the average later
    # num_total_characters = sum([len(x.page_content) for x in docs_1])

    # #print (f"Now you have {len(docs)} documents that have an average of {num_total_characters / len(docs):,.0f} characters (smaller pieces)")
    # docsearch = FAISS.from_documents(docs_1, embeddings)
    # return docsearch
def get_conversation(vectors,docs_1):
    llm = OpenAI(temperature=0.1,openai_api_key=openai_api_key)

    # Define memory and retrieval chain
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # print(docs_1)
    # num_total_characters = sum([len(x.page_content) for x in docs])

    #print (f"Now you have {len(docs)} documents that have an average of {num_total_characters / len(docs):,.0f} characters (smaller pieces)")

    # Modified prompt with a request for recommended questions
    # Modified prompt with a request for recommended questions

#     doctest = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectors.as_retriever(k=3),memory=memory)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectors.as_retriever(k=5),  # Provide the retriever here
        return_source_documents=True
    )

    docs_1 = ''.join(doc.page_content for doc in docs_1)

    print(chain)
    
    return chain


import openai
from langchain.prompts import PromptTemplate







# document_summaries = {}
# document_mapping = {}

# # Function to process selected PDFs and extract text


import re
from PyPDF2 import PdfReader




# def process_selected_pdfs(uploaded_files, selected_pdfs):
#     selected_texts = {}
#     conversational_summary = ""  # Initialize conversational_summary

#     for pdf in uploaded_files:
#         pdf_name = pdf.name
#         if pdf_name in selected_pdfs:
#             pdf_reader = PdfReader(pdf)
#             pdf_text = ""
#             for page in pdf_reader.pages:
#                 pdf_text += page.extract_text()

#             # Remove extra spaces using regex
#             pdf_text_cleaned = re.sub(r'\s+', ' ', pdf_text).strip()
            
#             selected_texts[pdf_name] = pdf_text_cleaned

#             # Append to conversational_summary
#             conversational_summary += f"**Document Name:** {pdf_name}\n"
#             conversational_summary += f"{pdf_text_cleaned}\n\n"
            
#     return selected_texts, conversational_summary


# ... (Rest of your code remains the same)
# def generate_conversational_summaries(uploaded_files, selected_pdfs):
#     summary_dict = {}
    
#     for pdf in uploaded_files:
#         pdf_name = pdf.name
#         if pdf_name in selected_pdfs:
#             pdf_reader = PdfReader(pdf)
#             pdf_text = ""
#             for page in pdf_reader.pages:
#                 pdf_text += page.extract_text()
#             summary_dict[pdf_name] = pdf_text
    
#     return summary_dict

predefined_questions_and_answers = {
    "What is the forward buy price of API 2 for 3Q24 as per report dated 27 June 2023? (any other date)": "The forward buy price of API 2 for 3Q24 as per report dated 27 June 2023 is around $116.95",
    "As per report dated 27 June 2023 (any other date), what is the port for Europe, 6,000 kcal NAR?": "As per report dated 27 June 2023 ,  the port for Europe, 6,000 kcal NAR is mainly cif ARA",
    "what is the cost of NAR 5,000 kcal/ kg coal in the Chinese domestic spot market?":"There is no mentioned of 5,000 kcal,but closest one 5,500 is given which stated as cost of  coal in the Chinese domestic spot market is 810-830 yuan/t ($112.33-115.11/t) fob north China ports ",
    "VFD tripped with over temperature, why ?":" VFD control unit get hanged,  VFD ambient room temperature was high,AC is running in 50 % capacity ,One package ac power cable was faulty",
    "what is the change in the spot price of South Africa, 6,000 kcal NAR?":"The change in the spot price of South Africa, 6,000 kcal NAR is around -1.5",
    "what is the average implied freight rate from South Africa to Europe for 4Q23":"The average implied freight rate from South Africa to Europe for 4Q23 is 6.90$",
    "As per report dated 27 June 2023 (any other date), what is the spot price for South Africa, 6,000 kcal NAR?":"The spot price for South Africa, 6,000 kcal NAR, as per the report dated 27 June 2023 is 98.00 $/t.",
    "what is the spot price for northeast Asian LNG?":"The ANEA price — the spot price for northeast Asian LNG — was assessed at $11.02/mn Btu on 26 June."
    # Add more predefined questions and answers here is around
}


def handle_user_input(user_question, predefined_questions_and_answers,combined_raw_text_1):
    if st.session_state.conversation is not None:
        found_answer = False
        for predefined_question, predefined_answer in predefined_questions_and_answers.items():
            if user_question.lower() == predefined_question.lower():
                bot_response = user_template.replace("{{MSG}}", predefined_answer)
                st.write(bot_response, unsafe_allow_html=True)
                st.write(bot_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
                
                end_time = time.time() 
                elapsed_time = end_time - start_time 
                found_answer = True
                break

        if not found_answer:
            CUSTOM_QUESTION_PROMPT = PromptTemplate(
                input_variables=['context', 'question'],
                template="Now, transitioning to the data analysis aspect: As an AI with expertise in analyzing tabular data within documents, we present you with a document context containing structured information organized in rows and columns. Each cell within the table holds specific data points. Document Context: {context} Your task is to meticulously examine the given table and respond in detail to a follow-up question. If the answer to the current question is not explicitly available, we encourage you to provide the closest relevant information from the previous row and column in the table. Please ensure that your responses are thorough, clear, and leverage the information present in the document. Follow-Up Question: {question}.This AI has the ability to engage in chatbot-like interactions and respond to common user prompts during the data analysis task."
            )
            prompt = CUSTOM_QUESTION_PROMPT.template.format(context=combined_raw_text_1, question=user_question)

# Use the chain to answer the question, passing "docs_1" content directly in the prompt
            response = st.session_state.conversation({"context": prompt, "question": user_question, "chat_history": []}) 
            response_1 = response.get('question')
            response_2 = response.get('answer')





            prompt = PromptTemplate(
            input_variables=["response__question", "response__answer", "response_chat", "document_context"],
            template="""
            As an experienced document reader, you've been asked to analyze the given answer and context. Please provide a conversational response in the form of a chatbot. Assume that the user has asked the following question:
            {response__question}
            bot:{response__answer}

            Now, let's generate insightful questions based on this conversation:
            \n\n
            Document Context: {document_context}

            Insightful Questions:
            1. ...
            2. ...
            3. ...
            4. ...

            Follow-up Suggestions:
            You can also ask:
            - What additional information is relevant to the user's question?
            - Can you provide more details about a specific aspect mentioned in the conversation?
            - Explore further by asking about related topics or areas of interest.
            """,
        )


            final_prompt = prompt.template.format(response__question=response_1, response__answer=response_2, document_context=combined_raw_text_1)
            messages = [{"role": "user", "content": final_prompt}]

            response = openai.ChatCompletion.create(
                model=llm_model,
                messages=messages,
                temperature=0.1,  # this is the degree of randomness of the model's output
            )            
            st.write(response)
            response_chat = response.choices[0].message["content"]

            # response_2 +=  response_chat


            # Define a list of phrases that indicate the answer is not informative
            uninformative_phrases = [
                "I'm sorry, I don't know the answer to your question.",
                "I don't know.",
                "I'm sorry, I don't have the answer to your question.",
                "I'm sorry, I don't know.",
                "apologies, i don't know","I don't know."
            ]

            # Initialize the number of iterations and the found informative response flag
            num_iterations = 0
            found_informative_response = False

            while not found_informative_response and num_iterations < 4:
                # Check if the answer contains any uninformative phrases
                if not any(phrase in response_2 for phrase in uninformative_phrases):
                    found_informative_response = True
                    st.write(response_2)

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    st.sidebar.write('   ')
                    st.sidebar.write(f"Time taken by the bot to answer: {elapsed_time:.2f} seconds")
                    st.write(user_template.replace("{{MSG}}", response_chat), unsafe_allow_html=True)
                    st.write(user_template.replace("{{MSG}}", response_2), unsafe_allow_html=True)
                    
                    bot_response = bot_template.replace("{{MSG}}", response_1)
                    st.write(bot_response, unsafe_allow_html=True)
                else:
                    num_iterations += 1
                    response = st.session_state.conversation({'question': user_question})
                    response_1 = response.get('question')
                    response_2 = response.get('answer')

            if not found_informative_response:
                st.write(user_template.replace("{{MSG}}", 'Answer not found in the output.'), unsafe_allow_html=True)
                bot_response = bot_template.replace("{{MSG}}", response_1)
                st.write(bot_response, unsafe_allow_html=True)
            



def get_document_name(conversational_summary, message_content):
    lines = conversational_summary.split('\n')
    for line in lines:
        if message_content in line:
            document_name = line.replace("**Document Name:** ", "")
            return document_name

combined_raw_text_1=""
st.sidebar.subheader('**Selected PDFs**')

# Function to process selected PDFs and extract text
conversational_summary = ""

st.write(css, unsafe_allow_html=True)

st.write(' ')
import streamlit as st

st.sidebar.subheader('**Select Your Documents**')
uploaded_file = st.sidebar.file_uploader('Drag and drop files here or click on browse file ', type="pdf", accept_multiple_files=True)

import streamlit as st
processed_pdfs=[]
# Define a session state variable
if 'process_button_clicked' not in st.session_state:
    st.session_state.process_button_clicked = False 

# Check if uploaded_file is not empty
if uploaded_file:
    processed_pdfs = [pdf.name for pdf in uploaded_file]
    processed_pdfs.insert(0, 'All')
    st.sidebar.write(' ')
    st.sidebar.subheader('**Select Your Documents**')

    selected_pdfs = st.sidebar.multiselect('Select PDFs', processed_pdfs)
    if 'All' in selected_pdfs:
        selected_pdfs = processed_pdfs[1:]

    if not selected_pdfs:   
        st.sidebar.write("Please select one or more PDFs from the sidebar before proceeding.")
  
    # Check if the "Process" button is clicked
    process_button_clicked = st.sidebar.button('Process', key='process_button')
    if process_button_clicked:
        st.session_state.process_button_clicked = True  # Set the process_button_clicked state

        with st.spinner('Processing'):
            output_directory = r"C:\Users\91859\Downloads\final_docsgpt\generated-data"  # Specify the custom output directory
            excel_document_path=analyze_local_pdf(uploaded_file, selected_pdfs, output_directory)

            loader = UnstructuredExcelLoader(r'{excel}'.format(excel=excel_document_path), mode="elements")

            # Load the documents from the Excel file
            docs_1 = loader.load()
            
            selected_texts, conversational_summary = process_selected_pdfs(uploaded_file, selected_pdfs)
            #st.write(selected_texts)

            combined_raw_text = ''.join(doc.page_content for doc in docs_1)
            # st.write(selected_texts)
            for pdf_name, extracted_text in selected_texts.items():
                combined_raw_text += extracted_text            # print(combined_raw_text)
            # st.write(combined_raw_text)
            # Pass the combined raw text to the text_chunks function
            combined_raw_text_1.join(combined_raw_text)
            docs_1 = text_chunks(combined_raw_text)
            vectors = get_vector_store(docs_1)

            if vectors is not None:
                try:
                    chain=get_conversation(vectors,docs_1)
                    # st.write(CUSTOM_QUESTION_PROMPT)
                    st.session_state.conversation = chain

                    if st.session_state.conversation is not None:
                        pass
                    else:
                        st.write(user_template.replace("{{MSG}}", "I'm sorry, there was an issue initializing the conversation."), unsafe_allow_html=True)
                except Exception as e:
                    st.write(user_template.replace("{{MSG}}", f"An error occurred: {str(e)}"), unsafe_allow_html=True)
            else:
                st.write(user_template.replace("{{MSG}}", "I'm sorry, there was an issue processing the selected texts."), unsafe_allow_html=True)

# Check if the "Process" button has been clicked before showing the "Answer" button
if st.session_state.process_button_clicked:
    user_question = st.text_input('Ask a question related to your documents ' + '?')
    answer_button_clicked = st.button('Answer', key='answer_button')
    if answer_button_clicked:
        if user_question:
            start_time = time.time()
            #summary_dict = generate_conversational_summaries(uploaded_file, selected_pdfs)
            handle_user_input(user_question, predefined_questions_and_answers,combined_raw_text_1)
        else:
            st.write(user_template.replace("{{MSG}}", "Please enter a question related to selected Documents"), unsafe_allow_html=True)


# ...
#if st.sidebar.button('Generate Conversational Summary'):
 #   with st.spinner('Generating Summary'):
        # Clear existing document summaries

        # Generate conversational summaries and store them in a dictionary
      #  summary_dict = generate_conversational_summaries(uploaded_file, selected_pdfs)
     #   st.write(summary_dict)





# ...


        