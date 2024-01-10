import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from pathlib import Path
import random
import time
import openai
import os
import uuid
from io import StringIO
from langchain.agents import AgentType, Tool, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from career_advisor import ChatController
from mock_interview import InterviewController
from callbacks.capturing_callback_handler import playback_callbacks
from basic_utils import convert_to_txt, read_txt, retrieve_web_content, html_to_text
from openai_api import check_content_safety
from dotenv import load_dotenv, find_dotenv
from common_utils import  check_content, get_generated_responses, get_web_resources
from openai_api import get_completion
import asyncio
import concurrent.futures
import subprocess
import sys
from multiprocessing import Process, Queue, Value
import pickle
import requests
from functools import lru_cache
from typing import Any
import multiprocessing as mp
from langchain.embeddings import OpenAIEmbeddings
from langchain_utils import retrieve_faiss_vectorstore, create_vectorstore, merge_faiss_vectorstore, create_vs_retriever_tools, create_retriever_tools
from pynput.keyboard import Key, Controller
from pynput import keyboard
import sounddevice as sd
from sounddevice import CallbackFlags
import soundfile as sf
import numpy  as np# Make sure NumPy is loaded before it is used in the callback
assert np  # avoid "imported but unused" message (W0611)
import tempfile
import openai
# from elevenlabs import generate, play, set_api_key
from time import gmtime, strftime
from playsound import playsound
from streamlit_modal import Modal
import json
from threading import Thread
from langchain.tools import ElevenLabsText2SpeechTool
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import re
import base64






_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
# set_api_key(os.environ["11LABS_API_KEY"])
save_path = os.environ["SAVE_PATH"]
temp_path = os.environ["TEMP_PATH"]
placeholder = st.empty()
# sd.default.samplerate=48000
sd.default.channels = 1, 2
sd.default.device = 1
duration = 600 # duration of each recording in seconds
fs = 44100 # sample rate
channels = 1 # number of channel
# COMBINATION = {keyboard.Key.r, keyboard.Key.ctrl}
device = 4
# keyboard = Controller()
# keyboard_event = Keyboard()



class Interview():

    userid: str=""
    COMBINATION = [{keyboard.KeyCode.from_char('r'), keyboard.Key.space}, {keyboard.Key.shift, keyboard.Key.esc}, {keyboard.KeyCode.from_char('s'), keyboard.Key.space}]
    currently_pressed = set()
    # placeholder = st.empty()
    q = queue.Queue()
    ctx = get_script_run_ctx()
    

    def __init__(self):
        self.set_png_as_page_bg('./static/styles/background.png')
        self._create_interviewbot()
        # self.thread_run()
        # thread = threading.Thread(target=self._create_interviewbot)
        # add_script_run_ctx(thread)
        # thread.start()


    @st.cache_data()
    def set_png_as_page_bg(_self, png_file):

        """
        function to display png as bg
        ----------
        png_file: png -> the background image in local folder
        """

        main_bg_ext = "png"
        
        st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(png_file, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


    # def thread_run(self):

    #     with ThreadPoolExecutor(max_workers=60) as executor:
    #         # ctx = get_script_run_ctx()
    #         # futures = [executor.submit(self._create_interviewbot, ctx)]
    #         # for future in as_completed(futures):
    #         #     yield future.result()
    #         future = executor.submit(self._create_interviewbot)
    #         future.result()


    def _create_interviewbot(self):



        with placeholder.container():

            with st.sidebar:
                
                add_vertical_space(3)
        
                st.markdown('''
                            
                How the mock interview works: 
     
                - refresh the page to start a new session   
                - press R + Space to start recording
                - press S + Space to stop recording
                - press Shift + Esc to end the session
                            
                ''')

                add_vertical_space(5)
                st.markdown('''
        
                Troubleshooting:

                1. if the AI cannot hear you, make sure your mic is turned on and enabled
                2. you can switch to the text only session by clicking on the button below
            
                            ''')
 
                switch = st.button("switch to text only session", key="switch_button")
                #temporary button
                feedback =st.button("end session", key="feedback", on_click=self.interview_feedback)
            

            if "userid" not in st.session_state:
                st.session_state["userid"] = str(uuid.uuid4())
                print(f"Session: {st.session_state.userid}")
                self.userid = st.session_state.userid


            if "interview_session_id" not in st.session_state:
                st.session_state["interview_session_id"] = str(uuid.uuid4())
                print(f"INTERVIEW Session: {st.session_state.interview_session_id}")
                modal = Modal(title="Welcome to your mock interview session!", key="popup")
                with modal.container():
                    with st.form( key='my_form', clear_on_submit=True):
                        add_vertical_space(1)
                        # st.markdown("Please fill out the form below before we begin")

                        st.text_area("tell me about your interview", placeholder="for example, you can say, my interview is with ABC for a store manager position", key="interview_about")

                        # st.text_input("links (this can be a job posting)", "", key = "interview_links", )

                        st.file_uploader(label="Upload your interview material or resume",
                                                        type=["pdf","odt", "docx","txt", "zip", "pptx"], 
                                                        key = "interview_files",
                                                        accept_multiple_files=True)
                        add_vertical_space(1)
                        st.form_submit_button(label='Submit', on_click=self.form_callback)  
    

            else:  
                
                # initialize submitted form variables
                if "about" not in st.session_state:
                    st.session_state["about"]=""
                if "job_posting" not in st.session_state:
                    st.session_state["job_posting"] = ""
                if "resume_file" not in st.session_state:
                    st.session_state["resume_file"] = ""
                # initialize keyboard listener
                if "listener" not in st.session_state:
                    self.file=None
                    self.record=True
                    new_listener = keyboard.Listener(
                    on_press=self.on_press,
                    on_release=self.on_release)
                    st.session_state["listener"] = new_listener
                # initialize backup text session
                if "text_session" not in st.session_state:
                    st.session_state["text_session"] = False
                # initialize main session interview agents
                if "baseinterview" not in st.session_state:
                    # update interview agents prompts from form variables
                    if  st.session_state.about!="":
                        additional_prompt_info = self.update_prompt(about=st.session_state.about, job_posting=st.session_state.job_posting, resume_file=st.session_state.resume_file)
                    else:
                        additional_prompt_info = ""
                    new_interview = InterviewController(st.session_state.userid, additional_prompt_info)
                    st.session_state["baseinterview"] = new_interview     
                    # welcome_msg = "Welcome to your mock interview session. I will begin conducting the interview now. Please review the sidebar for instructions. "
                    # message(welcome_msg, avatar_style="initials", seed="AI_Interviewer", allow_html=True)

                try:
                    self.new_interview = st.session_state.baseinterview  
                    self.listener = st.session_state.listener
                    try:
                        self.listener.start()
                    # RuntimeError: threads can only be started once  
                    except RuntimeError as e:
                        pass
                except AttributeError as e:
                    # if for some reason session ended in the middle, may need to do something different from raise exception
                    raise e
                # make directory for session recordings
                try: 
                    audio_dir =  f"./tmp_recording/{st.session_state.interview_session_id}/"
                    os.mkdir(audio_dir)
                except FileExistsError:
                    pass

                if switch:
                    st.session_state["text_session"] = True
                if st.session_state.text_session:
                    self.text_session()
        
                # self.listener.join()

                
     

    def on_press(self, key):

        """ Listens when a keyboards is pressed. """

        print("listener key pressed")
        if any([key in comb for comb in self.COMBINATION]):
            self.currently_pressed.add(key)
        if self.currently_pressed == self.COMBINATION[0]:
            print("on press: recording")
            filename = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            self.file = f"./tmp_recording/{self.userid}/{filename}.wav"
            thread = threading.Thread(target = self.record_audio2)
            add_script_run_ctx(thread, self.ctx)
            thread.start()
            # self.record_audio2()
        if self.currently_pressed == self.COMBINATION[1]:
            self.listener.stop()
            print("on press: quitting")
            thread = threading.Thread(target=self.interview_feedback)
            add_script_run_ctx(thread, self.ctx)
            thread.start()
        if self.currently_pressed == self.COMBINATION[2]:
            self.record = False
            print("on press: stopping")
            try:
                with open(self.file) as f:
                    f.flush()
                    f.close()
            except RuntimeError as e:
                raise e
            print("Recorded file closed")
            user_input = self.transcribe_audio2()
            response = self.new_interview.askAI(user_input)
            print(response)
            # tts = ElevenLabsText2SpeechTool()
            # speech_file = tts.run(response)
            # tts.play(speech_file)
            # st.session_state.tts.play(response)
            self.record = True

         
    def on_release(self, key):

        """ Listens when a keyboard is released. """
        
        try:
            self.currently_pressed.remove(key)
        except KeyError:
            pass
            

    def record_audio2(self):

        """ Records audio and write it to file. """

        def callback(indata, frame_count, time_info, status):
            self.q.put(indata.copy())

        with sf.SoundFile(self.file, mode='x', samplerate=fs,
                channels=channels) as file:
            with sd.InputStream(samplerate=fs, device=device,
                        channels=channels, callback=callback):
                while self.record:
                    print('recording')
                    file.write(self.q.get())


    # inspired by: https://github.com/VRSEN/langchain-agents-tutorial
    def transcribe_audio2(self) -> str:

        """ Sends audio file to OpenAI's Whisper model for trasncription and response """
        try:
            with open(self.file, "rb") as audio_file:
            # with open(temp_audio.name, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                print(f"Successfully transcribed file from openai whisper: {transcript}")
            # os.remove(temp_audio.name)
        except Exception as e:
            raise e
        return transcript["text"].strip()

    
    # def play_generated_audio(self, text, voice="Bella", model="eleven_monolingual_v1"):

    #     """ Deploys Eleven Labs for AI generated voice playback """

    #     audio = generate(text=text, voice=voice, model=model)
    #     play(audio)

    def interview_feedback(self):

        """ Provides interview session feedback as a downloadable file to the user. """

        with placeholder.container():
            modal = Modal(key="feedback_popup", title="Thank you for your participation in the interview session. I have a printout of your session summary and I value your feedback too!")
            with modal.container():
                feedback_path = self.new_interview.retrieve_feedback( )
                with open(feedback_path) as f:
                    st.download_button('Download my session summary', f)  # Defaults to 'text/plain'
                with st.form(key="feedback_form", clear_on_submit=True):
                    submit = st.form_submit_button()
                    #TODO write feedback for AI to a place



    def form_callback(self):

        """ Processes form information during form submission callback. """


        #TODO this progress spinner does not work yet
        placeholder = st.empty()     
        with placeholder.container():
            st.spinner("Please wait...")

        try: 
            vs_dir=f"./faiss/{st.session_state.userid}/"
            temp_dir = temp_path+st.session_state.userid
            user_dir = save_path+st.session_state.userid
            os.mkdir(temp_dir)
            os.mkdir(user_dir)
            os.mkdir(vs_dir)
        except FileExistsError:
            pass
        try:
            files = st.session_state.interview_files 
            self.process_file(files)
        except Exception:
            pass
        # try:
        #     link = st.session_state.interview_links
        #     self.process_link(link)
        # except Exception:
        #     pass 
        try:
            about = st.session_state.interview_about
            self.process_about_interview(about)
          
        except Exception:
            pass


    def text_session(self):

        """ Creates text only interview session. """

        # stop the keyboard listener
        try:
            st.session_state.listener.stop()
        except Exception:
            pass


        styl = f"""
        <style>
            .stTextInput {{
            position: fixed;
            bottom: 3rem;
            }}
        </style>
        """
        st.markdown(styl, unsafe_allow_html=True)
    
        with placeholder.container():

            # if 'interview_responses' not in st.session_state:
            #     st.session_state['interview_responses'] = list()
            # if 'interview_questions' not in st.session_state:
            #     st.session_state['interview_questions'] = list()

            if 'interview_response' not in st.session_state:
                st.session_state['interview_response'] = ""
            if 'interview_question' not in st.session_state:
                st.session_state['interview_question'] = ""

            col1, col2= st.columns(2)
            # with col1:
            #     if "question_container" not in st.session_state:
            #         st.session_state["question_container"] = st.container()
            # with col2:
            #     if "response_container" not in st.session_state:
            #         st.session_state["response_container"] = st.container()
            # response_container = st.container()

            if 'responseInput' not in st.session_state:
                st.session_state.responseInput = ''
            # def submit():
            #     st.session_state.responseInput = st.session_state.interview_input
            #     st.session_state.interview_input = ''    
            # # User input
            # ## Function for taking user provided prompt as input
            # def get_text():
            #     st.text_input("Your response: ", "", key="interview_input", on_change = submit)
            #     return st.session_state.responseInput
            # ## Applying the user input box
            # with response_container:
            #     user_input = get_text()
            #     response_container = st.empty()
            #     st.session_state.responseInput='' 


            # if user_input:

            #     # res = question_container.container()
            #     # streamlit_handler = StreamlitCallbackHandler(
            #     #     parent_container=res,
            #     #     # max_thought_containers=int(max_thought_containers),
            #     #     # expand_new_thoughts=expand_new_thoughts,
            #     #     # collapse_completed_thoughts=collapse_completed_thoughts,
            #     # )
            #     user_answer = user_input
            #     # answer = chat_agent.run(mrkl_input, callbacks=[streamlit_handler])
            #     ai_question = self.new_interview.askAI(user_answer, callbacks = None)
            #     st.session_state.interview_questions.append(ai_question)
            #     st.session_state.interview_responses.append(user_answer)
            # if st.session_state['interview_responses']:
            #     for i in range(len(st.session_state['interview_responses'])):
            #         with col1:
            #             message(st.session_state['interview_questions'][i], key=str(i), avatar_style="initials", seed="AI_Interviewer", allow_html=True)
            #         with col2:
            #             message(st.session_state['interview_responses'][i], is_user=True, key=str(i) + '_user',  avatar_style="initials", seed="Yueqi", allow_html=True)
            if st.session_state["interview_response"]:
                with col1:
                    message(st.session_state.interview_question)
                with col2:
                    message(st.session_state.interview_response)

            st.text_input("Your response: ", "", key="interview_input", on_change = self.chatbox_callback)


    def chatbox_callback(self):

        """ Processes user input from chatbox and prefilled question selection after submission. """
          
        response = self.new_interview.askAI(st.session_state.interview_input, callbacks = None)
        # st.session_state.interview_responses.append(st.session_state.interview_input)
        # st.session_state.interview_questions.append(response)
        st.session_state.interview_response = st.session_state.interview_input
        st.session_state.interview_question = response
        st.session_state.responseInput = st.session_state.interview_input
        st.session_state.interview_input = ''    


        

    

    def update_prompt(self, about: str, job_posting: str, resume_file:str) -> str:

        """ Updates prompts of interview agent and grader before initialization. 

        Args:

            about (str): preprocessed user's about interview input

            job_posting (str): job posting file path

            resume_file (str): resume file path  

        Retunrs:

            a concatenated string o f additional information such as company description, job specification found in the inputs. 
        
        """

        print(f"about: {about}")
        print(f"job psoting: {job_posting}")
        additional_interview_info = about
        try:
            resume_content = read_txt(resume_file)
        except Exception:
            resume_content = ""
        generated_dict=get_generated_responses(about_me=about, posting_path=job_posting, resume_content = resume_content)
        job = generated_dict.get("job", "")
        job_description=generated_dict.get("job description", "")
        company_description = generated_dict.get("company description", "")
        job_specification=generated_dict.get("job specification", "")
        # resume_field_names = generated_dict.get("field names", "")
        if job!=-1:
            # get top n job interview questions for this job
            query = f"top 10 interview questions for {job}"
            response = get_web_resources(query)
            additional_interview_info += f"top 10 interview questions for {job}: {response}"
        # if resume_field_names!="":
        #     for field_name in resume_field_names:
        #         additional_interview_info += f"""applicant's {field_name}: {generated_dict.get(field_name, "")}"""
        if job_description!="":
            additional_interview_info += f"job description: {job_description} \n"
        if job_specification!="":
            additional_interview_info += f"job specification: {job_specification} \n"
        if company_description!="":
            additional_interview_info += f"company description: {company_description} \n"

        return additional_interview_info

    def process_about_interview(self, about_interview:str) -> None:

        """ Processes user's about interview text input, including any links in the description."""
        
        about_interview_summary = get_completion(f"""Summarize the following description, if provided, and ignore all the links: {about_interview} \n
            If you are unable to summarize, ouput -1 only. Remember, ignore any links and output -1 if you can't summarize.""")
        if "about" not in st.session_state:
            st.session_state["about"] = about_interview_summary
        # process any links in the about me
        urls = re.findall(r'(https?://\S+)', about_interview)
        print(urls)
        if urls:
            for url in urls:
                self.process_link(url)

    def process_file(self, uploaded_files: Any) -> None:

        """ Processes user uploaded files including converting all format to txt, checking content safety, and categorizing content type  """

        for uploaded_file in uploaded_files:
            file_ext = Path(uploaded_file.name).suffix
            filename = str(uuid.uuid4())+file_ext
            tmp_save_path = os.path.join(temp_path, st.session_state.userid, filename)
            with open(tmp_save_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            end_path =  os.path.join(save_path, st.session_state.userid, Path(filename).stem+'.txt')
            # Convert file to txt and save it to uploads
            convert_to_txt(tmp_save_path, end_path)
            content_safe, content_type, content_topics = check_content(end_path)
            print(content_type, content_safe, content_topics) 
            if content_safe:
                # EVERY CONTENT TYPE WILL BE USED AS INTERVIEW MATERIAL
                self.update_vectorstore(content_type, end_path)
                if content_type=="resume":
                    print(f"user uploaded a resume file")
                    if "resume_file" not in st.session_state:
                        st.session_state["resume_file"]=end_path
                if content_type=="job posting":
                    print(f"user uploaded job posting")
                    if "job_posting" not in st.session_state:
                        st.session_state["job_posting"]= end_path
            else:
                print("file content unsafe")
                os.remove(end_path)

        

    def process_link(self, link: Any) -> None:

        """ Processes user shared links including converting all format to txt, checking content safety, and categorizing content type """

        end_path = os.path.join(save_path, st.session_state.userid, str(uuid.uuid4())+".txt")
        if html_to_text([link], save_path=end_path):
            content_safe, content_type, content_topics = check_content(end_path)
            if content_safe:
                if content_type=="browser error":
                    st.write("Link content cannot be parsed, please try another link.")
                 # EVERY CONTENT TYPE WILL BE USED AS INTERVIEW MATERIAL
                else:
                    self.update_vectorstore(content_type, end_path)
                    if content_type == "job posting":
                        print(f"user uploaded job posting")
                        if "job_posting" not in st.session_state:
                            st.session_state["job_posting"]= end_path
                    elif content_type=="resume":
                        print(f"user uploaded a resume file")
                        if "resume_file" not in st.session_state:
                            st.session_state["resume_file"]=end_path
            else:
                print("link content unsafe")
                os.remove(end_path)

    
    def update_vectorstore(self, content_type:str, end_path:str) -> None: 

        """ Converts uploaded content to vector store.
         
          Args:

            content_type: one of the following ["resume", "cover letter", "user profile", "job posting", "personal statement", "other"]

            end_path: file_path   
            
        """

        if content_type!="other":
            vs_name = content_type.strip().replace(" ", "_")
            vs = create_vectorstore("faiss", end_path, "file", vs_name)
        else:
            vs_name = "interview_material"
            vs = merge_faiss_vectorstore(vs_name, end_path)
        vs.save_local(f"./faiss/{st.session_state.userid}/interview/{vs_name}")




  
# async def inputstream_generator(channels=1, **kwargs):
#     """Generator that yields blocks of input data as NumPy arrays."""
#     q_in = asyncio.Queue()
#     loop = asyncio.get_event_loop()

#     def callback(indata, frame_count, time_info, status):
#         loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

#     stream = sd.InputStream(callback=callback, channels=channels, **kwargs)
#     with stream:
#         while True:
#             indata, status = await q_in.get()
#             yield indata, status

# async def stream_generator(blocksize, *, channels=1, dtype='float32',
#                            pre_fill_blocks=10, **kwargs):
#     """Generator that yields blocks of input/output data as NumPy arrays.

#     The output blocks are uninitialized and have to be filled with
#     appropriate audio signals.

#     """
#     assert blocksize != 0
#     q_in = asyncio.Queue()
#     q_out = queue.Queue()
#     loop = asyncio.get_event_loop()

#     def callback(indata, outdata, frame_count, time_info, status):
#         loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))
#         outdata[:] = q_out.get_nowait()

#     # pre-fill output queue
#     for _ in range(pre_fill_blocks):
#         q_out.put(np.zeros((blocksize, channels), dtype=dtype))

#     stream = sd.Stream(blocksize=blocksize, callback=callback, dtype=dtype,
#                        channels=channels, **kwargs)
#     with stream:
#         while True:
#             indata, status = await q_in.get()
#             outdata = np.empty((blocksize, channels), dtype=dtype)
#             yield indata, outdata, status
#             q_out.put_nowait(outdata)


# async def print_input_infos(**kwargs):
#     """Show minimum and maximum value of each incoming audio block."""
#     async for indata, status in inputstream_generator(**kwargs):
#         if status:
#             print(status)
#         print('min:', indata.min(), '\t', 'max:', indata.max())


# async def wire_coro(**kwargs):
#     """Create a connection between audio inputs and outputs.

#     Asynchronously iterates over a stream generator and for each block
#     simply copies the input data into the output block.

#     """
#     async for indata, outdata, status in stream_generator(**kwargs):
#         if status:
#             print(status)
#         outdata[:] = indata

# async def main(**kwargs):
#     print('Some informations about the input signal:')
#     try:
#         await asyncio.wait_for(print_input_infos(), timeout=2)
#     except asyncio.TimeoutError:
#         pass
#     print('\nEnough of that, activating wire ...\n')
#     audio_task = asyncio.create_task(wire_coro(**kwargs))
#     for i in range(10, 0, -1):
#         print(i)
#         await asyncio.sleep(1)
#     audio_task.cancel()
#     try:
#         await audio_task
#     except asyncio.CancelledError:
#         print('\nwire was cancelled')

if __name__ == '__main__':

    advisor = Interview()
    # try:
    #     asyncio.run(main(blocksize=1024))
    # except KeyboardInterrupt:
    #     sys.exit('\nInterrupted by user')