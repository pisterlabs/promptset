import streamlit as st 
from pytube import YouTube
import os
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from TemplatesHTML import bot_template, user_template


# %% Download audio
def download_audio(url):
    #url = input("Introduce the YouTube URL:")
    try:
        if "https://www.youtube.com/" in url:
            try:
                yt = YouTube( 
                str(url)) 
            except Exception as ee:
                print(ee)

            # extract only audio 
            audio = yt.streams.filter(only_audio=True).first() 
        
            destination = '.'
        
            # download the file 
            out_file = audio.download(output_path=destination) 
        
            # save the file 
            base, ext = os.path.splitext(out_file) 
            new_file = (base + '.mp3').replace(" ", "_")
            os.rename(out_file, new_file) 
        
            # result of success 
            string_ = yt.title + " is being processed."
            return new_file, string_
        else:
            return "This is not a valid URL!"
    except Exception as ee:
        print(ee)

# %% Check audio size 
def check_audio_size(file_location):
    try:
        file_stats = os.stat(file_location)
        file_size = file_stats.st_size / (1024*1024) # In MB
        print(f"Audio file size is: {file_size:.2f} MB")
        return file_size
    except Exception as e:
        print(e)

# %% Transform audio to text
def mp3_to_text(file_name):
    client = openai.OpenAI()
    file_size = check_audio_size(file_name)
    if file_size < 20: 
        try:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=open(file_name, "rb"),
                language="en"
            )
            os.remove(file_name)
            return transcript.text
        except Exception as e:
            print(f'{e} Try again.')
    else:
        return "Audio size is larger than 20 MB please try with a shorter video"
    


# %% Summarize with GPT3.5
def summarize_text(text):
    client = openai.OpenAI()
    doc_content = text
    prompt = f"""   
        It is necessary to generate a detailed summary about the following text 
        that is extracted from a youtube video:
        '''
            {doc_content}
        '''
        Please return the summary in format of three bullets containing the following fields:
        '''
            "Video topic": ...
            "Summary": ...
            "Keywords": ...
        '''
    """
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt 
            }
        ],
        model="gpt-3.5-turbo-1106",
        #`response_format={"type": "json_object"},
        temperature=0.2,
    )
    return response.model_dump()['choices'][0]['message']['content']
# %% Text chunks
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
# %% Vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# %% Get conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# %% Handle user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
