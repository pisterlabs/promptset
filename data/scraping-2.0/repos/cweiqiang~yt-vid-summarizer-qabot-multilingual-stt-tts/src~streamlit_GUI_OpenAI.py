# https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
'''

KSP2: https://www.youtube.com/watch?v=SwKYq17W-_s&t=15s
KSP1: https://www.youtube.com/watch?v=YUpPDAqrf48
CIV5: https://www.youtube.com/watch?v=ZVqWCdKif3c
AOE4: https://www.youtube.com/watch?v=WjZiRvqjov8


completed tasks:
- implement langchain query
- implement translation
- implement audio output
- implement video embedding
- implement audio input
- fix the error in _last.mp3
- position the question input box at the top of QA section
- modularize the qa_bot() function
- added garbage collector for old .mp3 files
- fixed the problem with persistent text and audio input

remaining tasks:
- add docstring and type hints


'''
import re
import os
import random
import time
import streamlit as st
import textwrap
from gtts import gTTS
import base64
import textwrap
# from langchain.llms import OpenAI
# from langchain.llms import VertexAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.chat_models import ChatOpenAI
# from langchain.chat_models import ChatVertexAI
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import translators as ts

from dotenv import load_dotenv, find_dotenv

# read local .env file with OPENAI_API_KEY [Recommended]
# it searches for .env file in the current directory
# and loads the environment variables from it
# To run openAI API, you need to set up the OPENAI_API_KEY in .env file
_ = load_dotenv(find_dotenv())

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/waynechen/.config/gcloud/application_default_credentials.json'

api_key = os.getenv("OPENAI_API_KEY")
# For English video transcript
# e.g. OpenAI dev day : https://www.youtube.com/watch?v=qJp6JPRzQ_8
lang_dict = {
    'English': 'en',
    'ms-MY': 'ms',
    'zh': 'zh-CN',
    'zh-TW': 'zh-TW',
    "ta-SG": "ta",
    "ja-JP": "ja",
    "ko-KR": "ko",
    "yue-Hant-HK": "yue"
}
# For Selected video transcript in Chinese
# e.g. Chinese News https://www.youtube.com/watch?v=jeXalyWP7HE
# lang_dict = {
#     'English': 'en',
#     'ms-MY': 'ms',
#     'zh': "zh-Hans",
#     'zh-TW': "zh-Hant",
#     "ta-SG": "ta",
#     "ja-JP": "ja",
#     "ko-KR": "ko",
#     "yue-Hant-HK": "yue"
# }


def youtube_video_url_is_valid(url: str) -> bool:
    """
    Checks if the given URL is a valid YouTube video URL.

    Args:
        url (str): The YouTube video URL.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """

    pattern = r'^https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)(\&[a-zA-Z0-9_]+=[\w\d]+)*$'

    match = re.match(pattern, url)
    return match is not None


def obtain_transcript(url: str) -> str:
    """
    Obtains the transcript of a YouTube video.

    Args:
        url (str): The YouTube video URL.

    Returns:
        str: The transcript of the YouTube video.
    """
    try:
        # loader = YoutubeLoader.from_youtube_url(url, language="en-US")
        loader = YoutubeLoader.from_youtube_url(url,
                                                add_video_info=True,
                                                language=[
                                                    "en", "de", "nl", "sv", "da", "no", "fr", "es", "it",
                                                    "ms", "pt", 'zh-CN', 'zh-TW',
                                                    "zh-Hans", "zh-Hant", "ja", "ko", "ru", "pl", "cs", "tr", "th"],
                                                translation="en",
                                                )

        transcript = loader.load()

        return transcript  # full transcript is return

    except Exception as e:
        return f"Error while loading YouTube video and transcript: {e}"


def summarize(transcript: str) -> str:
    """
    Summarizes the given transcript.

    Args:
        transcript (str): The transcript to be summarized.

    Returns:
        str: The summary of the transcript.
    """
    try:
        llm = OpenAI(temperature=0.3,
                     model='gpt-3.5-turbo',
                     openai_api_key=api_key)
        # llm = VertexAI(temperature=0.3, max_output_tokens=512)
        prompt = PromptTemplate(
            template="""Summarize the youtube video whose transcript is provided within backticks \
            ```{text}```
            """, input_variables=["text"]
        )
        combine_prompt = PromptTemplate(

            template="""Combine all the youtube video transcripts  provided within backticks \
            ```{text}```
            Provide a summary within 400 words.
            """, input_variables=["text"]
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100000, chunk_overlap=50)
        text = text_splitter.split_documents(transcript)

        chain = load_summarize_chain(llm, chain_type="map_reduce",
                                     map_prompt=prompt, combine_prompt=combine_prompt)
        answer = chain.run(text)
    except Exception as e:
        return f"Error while processing and summarizing text: {e}"

    st.session_state["summary_en"] = answer.strip()

    return answer.strip()

# Main part of the Streamlit app


def get_file_download_link(filename: str) -> str:
    """
    Generates a download link for the specified file.

    Args:
        filename (str): The name of the file to be downloaded.

    Returns:
        str: The HTML code for the download link.
    """

    with open(filename, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href


def create_qa_retriever(transcript: str) -> RetrievalQA:
    """
    Creates a question-answering (QA) retriever based on the given transcript.

    Args:
        transcript (str): The transcript to create the QA retriever from.

    Returns:
        RetrievalQA: The QA retriever object.
    """

    # Initialize text splitter for QA
    text_splitter_qa = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)

    print("text_splitter_qa: ", text_splitter_qa)
    print("type(text_splitter_qa): ", type(text_splitter_qa))

    print("transcript: ", transcript)

    # Split text into docs for QA
    docs_qa = text_splitter_qa.split_documents(transcript)

    # Create the LLM model for the question answering
    # llm_question_answer = ChatVertexAI(temperature=0.2)
    llm_question_answer = ChatOpenAI(temperature=0.2,
                                     model='gpt-3.5-turbo',
                                     openai_api_key=api_key)

    # Create the vector database and RetrievalQA Chain
    # embeddings = VertexAIEmbeddings()
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(docs_qa, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=llm_question_answer, chain_type="stuff", retriever=db.as_retriever())

    return qa


def qa_bot(qa: RetrievalQA, language: str):
    """
    Runs the question-answering (QA) bot.

    Args:
        qa (RetrievalQA): The QA retriever object.
        language (str): The selected language for the bot.
    """

    # Get the new question
    col1, col2 = st.columns(2)
    with col1:
        question = st.text_input("Ask a question:")
    with col2:
        audio_data = audio_recorder()
        if audio_data is not None and (st.session_state['last_audio_data'] != audio_data):
            st.session_state['last_audio_data'] = audio_data
            # write bytestring to .wave file
            with open('audio.wave', 'wb') as f:
                f.write(audio_data)
            r = sr.Recognizer()
            # record .wave file into AudioData type
            with sr.AudioFile('audio.wave') as source:
                audio = r.record(source)
                print(audio)
                print(language)
            question = r.recognize_google(audio, language=language)

    # generate answer for the user's question
    if question and (st.session_state['last_question'] != question):

        # update the last question
        st.session_state['last_question'] = question

        # Run the QA chain to query the data
        if language != 'English':
            q_trans = question
            q_en = ts.translate_text(query_text=q_trans,
                                     from_language=lang_dict[language],
                                     to_language='en',
                                     translator='google')

            a_en = st.session_state['qa'].run(q_en)
            st.session_state.history.append({f'q_{lang_dict[language]}': q_trans,
                                             "q_en": q_en,
                                             "a_en": a_en})

        # everything in English
        else:
            q_en = question
            a_en = qa.run(q_en)
            st.session_state.history.append({"q_en": q_en, "a_en": a_en})

    # Display all Q&A
    for i, q_and_a in enumerate(st.session_state.history):
        if language != 'English' and (q_and_a.get('q_en') is not None):
            # if not already translated to the same language before
            if st.session_state.history[i].get(f'a_{lang_dict[language]}') is None:
                if st.session_state.history[i].get(f'q_{lang_dict[language]}') is None:
                    q_trans = ts.translate_text(query_text=q_and_a['q_en'],
                                                from_language='en',
                                                to_language=lang_dict[language],
                                                translator='google')
                else:
                    q_trans = q_and_a[f'q_{lang_dict[language]}']
                a_trans = ts.translate_text(query_text=q_and_a['a_en'],
                                            from_language='en',
                                            to_language=lang_dict[language],
                                            translator='google')
                st.session_state.history[i][f'q_{lang_dict[language]}'] = q_trans
                st.session_state.history[i][f'a_{lang_dict[language]}'] = a_trans

        if not os.path.exists(f"./data/q_audio_{lang_dict[language]}_{i}.mp3"):
            q_audio = gTTS(
                text=q_and_a[f'q_{lang_dict[language]}'], lang=lang_dict[language], slow=False)
            a_audio = gTTS(
                text=q_and_a[f'a_{lang_dict[language]}'], lang=lang_dict[language], slow=False)
            q_audio.save(f"./data/q_audio_{lang_dict[language]}_{i}.mp3")
            a_audio.save(f"./data/a_audio_{lang_dict[language]}_{i}.mp3")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<p style="color:green;">**Q{i}:** {q_and_a[f"q_{lang_dict[language]}"]}</p>', unsafe_allow_html=True)
            st.audio(
                f'./data/q_audio_{lang_dict[language]}_{i}.mp3', format='audio/mp3')

        # Add A to the right column
        with col2:
            st.markdown(
                f'<p style="color:black;">**A{i}:** {q_and_a[f"a_{lang_dict[language]}"]}</p>', unsafe_allow_html=True)
            st.audio(
                f'./data/a_audio_{lang_dict[language]}_{i}.mp3', format='audio/mp3')


def initialize_lang() -> str:
    """
    Initializes the selected language.

    Returns:
        str: The selected language.
    """
    # Dropdown for language selection
    selected_language = st.selectbox('Language:',
                                     options=list(lang_dict.keys()))

    # if button is clicked, show the selectbox
    if selected_language:
        st.session_state["language"] = selected_language
    else:
        st.session_state["language"] = 'English'

    return selected_language


def initialize_others(language: str):
    """
    Initializes other session states.

    Args:
        language (str): The selected language.
    """
    # Initialize qa in session state if it doesn't exist
    if 'qa' not in st.session_state:
        st.session_state['qa'] = None

    # Initialize qa in session state if it doesn't exist
    if 'summary_en' not in st.session_state:
        st.session_state['summary_en'] = None

    # Initialize translated summary in session state if it doesn't exist
    if f'summary_{lang_dict[language]}' not in st.session_state:
        st.session_state[f'summary_{lang_dict[language]}'] = None

    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = None

    if 'last_audio_data' not in st.session_state:
        st.session_state['last_audio_data'] = None

    # Remove .mp3 files in the current directory
    directory = './data'
    for file in os.listdir(directory):
        if file.endswith('.mp3'):
            file_path = os.path.join(directory, file)
            os.remove(file_path)


def translate_summary():
    """
    Translates the summary to the selected language.

    This function translates the summary from English to the selected language
    using the Google Translate API and stores the translated summary in the
    session state.

    Raises:
        Exception: Error while translating the summary.
    """
    language = st.session_state["language"]
    language_code = lang_dict[language]

    summary = st.session_state["summary_en"]

    if language != 'English':
        translated_summary = ts.translate_text(query_text=summary,
                                               from_language='en',
                                               to_language=language_code,
                                               translator='google')
        st.session_state[f"summary_{language_code}"] = translated_summary


def display_summary():
    """
    Displays the summary in the selected language.

    This function displays the summary in the selected language. If the selected
    language is not English, it retrieves the translated summary from the session
    state and displays it. If the selected language is English, it retrieves the
    original English summary from the session state and displays it.

    Raises:
        FileNotFoundError: Audio file for the translated summary not found.
    """
    language = st.session_state["language"]
    language_code = lang_dict[language]

    if language != 'English':
        st.markdown(st.session_state[f"summary_{language_code}"])

    elif st.session_state.get('summary_en') is not None:
        st.markdown(st.session_state["summary_en"])

    # audio out summary
    if st.session_state.get(f"summary_{language_code}") is not None:
        summary_audio = gTTS(
            text=st.session_state[f"summary_{language_code}"], lang=language_code, slow=False)
        summary_audio.save(f"./data/summary_{language_code}_audio.mp3")
        st.audio(
            f"./data/summary_{language_code}_audio.mp3", format='audio/mp3')


def main():
    """
    Main function to run the Streamlit application for YouTube video summarization.
    """
    st.title("Video Guru")

    url = st.text_input("Enter Youtube video URL here")

    if url != "":
        st.video(url)

    language = initialize_lang()

    if st.button("Summarize"):

        initialize_others(language)

        if not youtube_video_url_is_valid(url):
            st.error("Please enter a valid Youtube video URL.")
            return

        st.session_state.history = []

        # complete the code block before 'Summarizing...' Button status change
        with st.spinner("Summarizing..."):

            transcript = obtain_transcript(url)

            # print("Line 443: transcript: ", transcript)

            summarize(transcript)

            # create qa retriever for subsequent QA session
            st.session_state['qa'] = create_qa_retriever(transcript)

    # Translate the summary if the selected language is not English
    if language != 'English' and (st.session_state.get("summary_en") is not None):
        translate_summary()

    if st.session_state.get("summary_en") is not None:
        display_summary()

    # execute Q&A session
    if st.session_state.get('qa') is not None:
        qa_bot(st.session_state['qa'],
               language=st.session_state.get("language"))


if __name__ == "__main__":
    main()
