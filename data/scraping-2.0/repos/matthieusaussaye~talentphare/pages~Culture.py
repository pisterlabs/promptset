import streamlit as st
from streamlit_lottie import st_lottie
from typing import Literal
from dataclasses import dataclass
import json
import base64
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import nltk
from prompts.prompts import templates
# Audio
from speech_recognition.openai_whisper import save_wav_file, transcribe
from audio_recorder_streamlit import audio_recorder
#from aws.synthesize_speech import synthesize_speech
from IPython.display import Audio

### ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

st.markdown("#### Bienvenue, nous allons discuter des valeurs de notre entreprise")
jd = st.text_area("Culture de l'entreprise:")
st.session_state.culture_entreprise = jd
#jd="The Banque Cantonale du Valais (Switzerland) is a bank that deeply values innovation rooted in tradition, maintains close and meaningful relationships with its clients, is committed to sustainable operations and environmental responsibility, and upholds a high standard of professional competence and expertise in its services.The Banque Cantonale du Valais (Switzerland) is seeking a Data Analyst to join their DATA team in Sion, with a workload of 80-100%. The role involves interacting with users, analyzing their needs, and supporting them in using new tools. The Data Analyst will be expected to collaborate with business teams in design workshops, actively participate in technological developments related to data management, and write technical documentation. Ideal candidates should have higher education in computer science (or equivalent) and experience in a similar role. Knowledge of the banking sector is considered a plus. Proficiency in computer tools such as Power BI and Azure Databricks, as well as good writing skills and knowledge of German and/or English, are required. The candidate should be committed, proactive, passionate about their profession, and able to work autonomously and collaboratively with other experts."

#st.toast("4097 tokens is roughly equivalent to around 800 to 1000 words or 3 minutes of speech. Please keep your answer within this limit.")
### ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
@dataclass
class Message:
    """class for keeping track of interview history."""
    origin: Literal["human", "ai"]
    message: str

with st.sidebar:
    st.markdown("IDIAP Create Challenge 2023")


if "jd_history" in st.session_state:
    interview = []
    dict_to_save = {}
    st.info(str(len(st.session_state.jd_history)))
    if len(st.session_state.jd_history) == 9:
        for e in st.session_state.jd_history:
            interview.append(e)
        interview_str = '|'.join([str(item) for item in interview])
        st.session_state.interview = interview_str
        st.session_state.short_interview = interview_str
        dict_to_save["culture_entreprise"] = st.session_state.culture_entreprise
        dict_to_save["name"] = st.session_state.name_surname
        dict_to_save["interview"] = st.session_state.interview
        dict_to_save["short_interview"] = st.session_state.short_interview
        st.info(str(dict_to_save))
def save_vector(text):
    """embeddings"""

    nltk.download('punkt')
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)
     # Create emebeddings
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch

def initialize_session_state_jd():
    """ initialize session states """
    if 'jd_docsearch' not in st.session_state:
        st.session_state.jd_docserch = save_vector(jd)
    if 'jd_retriever' not in st.session_state:
        st.session_state.jd_retriever = st.session_state.jd_docserch.as_retriever(search_type="similarity")
    if 'jd_chain_type_kwargs' not in st.session_state:
        Interview_Prompt = PromptTemplate(input_variables=["context", "question"],
                                          template=templates.jd_template)
        st.session_state.jd_chain_type_kwargs = {"prompt": Interview_Prompt}
    if 'jd_memory' not in st.session_state:
        st.session_state.jd_memory = ConversationBufferMemory()
    # interview history
    if "jd_history" not in st.session_state:
        st.session_state.jd_history = []
        st.session_state.jd_history.append(Message("ai",
                                                   "Bonjour, je suis votre intervieweur aujourd'hui. Je vais vous poser quelques questions concernant les valeures de notre entreprise. Êtes-vous prêt à commencer l'entretien?"))
    # token count
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "jd_guideline" not in st.session_state:
        llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.8,)
        st.session_state.jd_guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type_kwargs=st.session_state.jd_chain_type_kwargs, chain_type='stuff',
            retriever=st.session_state.jd_retriever, memory = st.session_state.jd_memory).run("Vous êtes l'intervieweur pour un poste dans une entreprise. Ceci est le résumé des valeurs de l'entreprise. Pose une question relatif à ces valeurs pour tester le culture fit du candidat")
    # llm chain and memory
    if "jd_screen" not in st.session_state:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.8, )
        PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template="""Je veux que vous agissiez comme un intervieweur en suivant strictement la directive dans la conversation actuelle.

                        Posez-moi des questions et attendez mes réponses comme le ferait un humain. Ne rédigez pas d'explications.
                        Le candidat n'a pas accès à la directive.
                        Ne posez qu'une seule question à la fois.
                        N'hésitez pas à poser des questions de suivi si vous le jugez nécessaire.
                        Ne posez pas la même question.
                        Ne répétez pas la question.
                        Le candidat n'a pas accès à la directive.
                        Votre nom est GPTInterviewer.
                        Je veux que vous ne répondiez que comme un intervieweur.
                        Ne rédigez pas toute la conversation en une fois.
                        Le candidat n'a pas accès à la directive.
                        
                        Conversation actuelle :
                        {history}
                        
                        Candidat : {input}
                        IA :""")

        st.session_state.jd_screen = ConversationChain(prompt=PROMPT, llm=llm,
                                                           memory=st.session_state.jd_memory)
    if 'jd_feedback' not in st.session_state:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.8, )
        st.session_state.jd_feedback = ConversationChain(
            prompt=PromptTemplate(input_variables=["history", "input"], template=templates.feedback_template),
            llm=llm,
            memory=st.session_state.jd_memory,
        )

def answer_call_back():
    with get_openai_callback() as cb:
        # user input
        human_answer = st.session_state.answer
        # transcribe audio
        if voice:
            save_wav_file("temp/audio.wav", human_answer)
            try:
                input = transcribe("temp/audio.wav")
                # save human_answer to history
            except:
                st.session_state.jd_history.append(Message("ai", "Sorry, I didn't get that."))
                return "Please try again."
        else:
            input = human_answer

        st.session_state.jd_history.append(
            Message("human", input)
        )
        # OpenAI answer and save to history
        llm_answer = st.session_state.jd_screen.run(input)
        # speech synthesis and speak out
        #audio_file_path = synthesize_speech(llm_answer)
        # create audio widget with autoplay
        #audio_widget = Audio(audio_file_path, autoplay=True)
        # save audio data to history
        st.session_state.jd_history.append(
            Message("ai", llm_answer)
        )
        st.session_state.token_count += cb.total_tokens
        #return audio_widget

### ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# sumitted job description
if jd:
    # initialize session states
    initialize_session_state_jd()
    #st.write(st.session_state.jd_guideline)
    credit_card_placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        feedback = st.button("Get Interview Feedback")
    with col2:
        guideline = st.button("Show me interview guideline!")
    chat_placeholder = st.container()
    answer_placeholder = st.container()
    audio = None
    # if submit email adress, get interview feedback imediately
    if guideline:
        st.write(st.session_state.jd_guideline)
    if feedback:
        evaluation = st.session_state.jd_feedback.run("please give evalution regarding the interview")
        st.markdown(evaluation)
        st.download_button(label="Download Interview Feedback", data=evaluation, file_name="interview_feedback.txt")
        st.stop()
    else:
        with answer_placeholder:
            voice: bool = st.checkbox("Utiliser mon micro pour répondre")
            if voice:
                answer = audio_recorder(pause_threshold = 2.5, sample_rate = 44100)
                #st.warning("An UnboundLocalError will occur if the microphone fails to record.")
            else:
                answer = st.chat_input("Your answer")
            if answer:
                st.session_state['answer'] = answer
                audio = answer_call_back()
        with chat_placeholder:
            for answer in st.session_state.jd_history:
                if answer.origin == 'ai':
                    
                    if audio:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                            st.write(audio)
                    else:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                else:
                    with st.chat_message("user"):
                        st.write(answer.message)

        credit_card_placeholder.caption(f"""
        Progress: {int(len(st.session_state.jd_history) / 30 * 100)}% completed.""")
else:
    st.info("Merci de préciser la culture d'entreprise pour commencer l'entretien")
    if 'name_surname' in st.session_state:
        st.write(f"Nom: {st.session_state.name_surname}")
    else:
        st.write("Merci de préciser le nom à l'accueil")
