from dotenv import load_dotenv
import streamlit as st
import os
from helpers.s3helpers import S3helpers
from helpers.utilities import Utility
import boto3
import openai
from sources.website import Website
from sources.pdfword import PdfWord
from streamlit_chat import message
from llama_index import SimpleDirectoryReader, VectorStoreIndex

load_dotenv()
title = st.title(os.environ['APP_NAME'])
c1, c2 = st.columns(2)
#with st.sidebar:
#    openai_key = st.text_input('OpenAI API Key', key='chatbot_api_key')
#    sbtn = st.button('Save')
#    if sbtn:
#        openai.api_key = os.environ['OPENAI_API_KEY']

openai.api_key = os.environ['OPENAI_API_KEY']
s3 = boto3.resource('s3')
# Set up cloud and openai key using UI
# c1, c2, c3 = st.columns(3)
# with c3:
#     open_ai_key = st.text_input("OpenAI API Key")
#     cloud_box = st.selectbox('Select cloud service provider', ('AWS', 'GCP', 'Azure'))
#     if cloud_box == 'AWS':
#         access_key = st.text_input('AWS Access Key ID')
#         secret_key = st.text_input('AWS Secret Access Key')
#         sbutton = st.button('Save')
#         if sbutton:
#             openai.api_key = open_ai_key
#             cloud = AWS()
#             obj = cloud.setup(access_key=access_key, secret_key=secret_key)


with c1:
    option = st.selectbox('Select Data Source', ('PDF/Word', 'Website', 'Chat'))

    if option == 'PDF/Word':
        uploaded_files = st.file_uploader("Choose a PDF/Word file")
        if uploaded_files is not None:
            # Uploading the input file to S3 Bucket
            storage = S3helpers()
            storage.upload(file_obj=uploaded_files)
            # st.write('File Uploaded to S3.')
            storage.download(file_obj=uploaded_files, temp_folder='tmp')
            documents = SimpleDirectoryReader('tmp/').load_data()

            index = VectorStoreIndex.from_documents(documents)
            chat_engine = index.as_chat_engine()
            # query_engine = index.as_query_engine()
            ###################################################################
            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

            with st.form("chat_input", clear_on_submit=True):
                a, b = st.columns([4, 1])
                user_input = a.text_input(
                    label="Your message:",
                    placeholder="What would you like to say?",
                    label_visibility="collapsed",
                )
                b.form_submit_button("Send", use_container_width=True)

            for idx, msg in enumerate(st.session_state.messages):
                message(msg["content"], is_user=msg["role"] == "user", key=idx)
                # message(msg["content"], is_user=msg["role"], key=idx)

            if user_input and not os.environ['OPENAI_API_KEY']:
                st.info("Please add your OpenAI API key to continue.")

            pw = PdfWord()
            if user_input and os.environ['OPENAI_API_KEY']:
                openai.api_key = os.environ['OPENAI_API_KEY']
                st.session_state.messages.append({"role": "user", "content": user_input})
                message(user_input, is_user=True)
                response = pw.analyze(temp_dir='tmp', user_prompt_text=user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                message(response, is_user=False)

            # Deleting the temporary files and folders created. # CODE Under PROGRESS
            ut = Utility()
            ut.delete_tmp_folder(folder_name='tmp', file_name=uploaded_files.name)

            ###################################################################
            # PDF/Word Prompt Box
            # query = st.text_area('Enter your prompt query')
            # st.write(query)
            # Parsing and analyzing PDF/Word
            # analyze_profile = st.button('Analyze')

            # if analyze_profile:
            #     with st.spinner('Analyzing ...'):
            #         pw = PdfWord()
            #         result = pw.analyze(temp_dir='tmp', user_prompt_text=query)
                    # st.write(result)
                    # object = s3.Object(
                    #     bucket_name=os.environ['AWS_S3BUCKET_PROCESSED'],
                    #     key=os.environ['APP_NAME'] +'/'+ uploaded_files.name + '.txt'
                    # )
                    # object.put(Body=result)
                    # st.write(f'Saved {option} analysis..')

    if option == "Website":
        url_link = st.text_input('URL')
        query = st.text_area('Enter your prompt query')
        analyze_profile = st.button('Analyze')
        if analyze_profile:
            with st.spinner('Analyzing...'):
                li = Website()
                result = li.analyze(user_website_link=url_link, user_prompt_text=query)
                if result:
                    with c2:
                        st.write(result)
                else:
                    st.write('Something went wrong')


