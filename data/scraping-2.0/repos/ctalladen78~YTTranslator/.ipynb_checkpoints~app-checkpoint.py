import streamlit as st
import requests
import pandas as pd
import youtube_transcript_api
from deep_translator import GoogleTranslator
import openai
import langchain
from transformers import pipeline

#Title
st.title("Youtube CJK translator")

language = ['ja','ko','zh-CN']
selected_lang = language[0]

with st.sidebar:
    st.title('🍁 Translator App')
    st.markdown('''
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    Inspired by : [YT-TLDR](https://www.you-tldr.com)
    ''')
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    #     radio button to select japanese by default
    st.radio("Language selected:",
            ["Japanese","Chinese","Korean"])

home_tab, summary_tab, qna_tab = st.tabs(["Home", "Summary", "Q&A"])


summarizer = None
summary_text = ""
combined_text = ""

@st.cache_resource
def init_model():
    return pipeline('summarization',
                      model="t5-small",
                      # model_kwargs={"cache_dir": './models'}
                     )
@st.cache_data
def combine_text(txt):
    combined_text = ' '.join(txt)
    # summary_text = get_summarization()
    return
    
# @st.cache(allow_output_mutation=True)
@st.cache_data
def get_translation(yt_path):
    # translate japanese
    vid = yt_path.split("=")[1]

    transcript_list = youtube_transcript_api.YouTubeTranscriptApi.list_transcripts(vid)

    transcript = transcript_list.find_transcript([f"{selected_lang}"])
    translated_text_ls = []
    try:
        transcript_fetched = transcript.fetch()
        transcript_text = [item['text'] for item in transcript_fetched]
        combine_text(transcript_text)
        translator = GoogleTranslator(source=f"{selected_lang}", target='en')

        # for item in transcript_fetched
        for idx, item in enumerate(transcript_fetched):
            tr = translator.translate(str(item['text']))
            translated_text_ls.append({'start':str(item['start']),
                                   'duration':str(item['duration']),
                                   'text':str(item['text']),
                                   'translated':tr})
            print(translated_text_ls[idx])
    except:
        translated_text_ls = []
    return translated_text_ls

# @st.cache_resource()
# # summarize all translated transcript content
# def get_summarization_model(txt):
#     # request replicate api
#     pipeline = 
#     return 

# home
with home_tab:
    # summarizer = init_model()
    yt_path = st.text_input("Enter youtube link to translate...",
                        "https://www.youtube.com/watch?v=FiLHU4QiUs8")

    st.write("### YT Translation")
    if yt_path is not None:
        with st.spinner("Translating...."):
            translated_text_ls=get_translation(yt_path)
            translated_text_df = pd.DataFrame(translated_text_ls)
            translated_text_df.head(10)
            st.dataframe(translated_text_df)

            # try:
            #     translated_text_ls=get_translation()
            #     translated_text_df = pd.DataFrame(translated_text_ls)
            #     translated_text_df.head(10)
            #     st.dataframe(translated_text_df)
            # except:
            #     print("translation error")

# summary_tab
with summary_tab:
    st.write("### Summary")
    if yt_path is not None:
        with st.spinner("Summarizing...."):
            # summary_text = summarizer(combined_text)
            st.write(summary_text)


    # qna
#     st.write("### Q&A")
#     if prompt := st.chat_input():
#         if not openai_api_key:
#             st.info("Please add your OpenAI API key to continue.")
#             st.stop()

#         # prompt template
#         openai.api_key = openai_api_key
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         st.chat_message("user").write(prompt)
#         response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
