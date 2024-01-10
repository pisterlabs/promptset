import streamlit as st
from streamlit_chat import message
from utils import get_initial_message, get_chatgpt_response, get_chatgpt_response_stream_chunk, update_chat
import os
from dotenv import load_dotenv
import openai
from secret_openai_apikey import api_key
import anki_utils
from collections import defaultdict
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from random import randrange
import stt

from streamlit_bokeh_events import streamlit_bokeh_events

from gtts import gTTS
from io import BytesIO
import openai
os.environ['OPENAI_API_KEY'] = api_key 
load_dotenv()




def mic_button():
    return Button(label='🎙️', button_type='success')



def get_result(stt_button):
    result = streamlit_bokeh_events(
    bokeh_plot = stt_button,
    events="GET_TEXT,GET_ONREC,GET_INTRM",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)

    return result

def user_text_input_surrogate(tr, session_state_object, value=None):
        if value is None:
            tr.text_input("You: ", value=session_state_object['query'], placeholder='speak or type', label_visibility="collapsed")
        else:
            tr.text_input("You: ", value=value, placeholder='speak or type', label_visibility="collapsed")

def mic_button_monitor_surrogate(tr, stt_button, session_state):
    stt_button.js_on_event('button_click', get_mic_button_js()) # second argument is a callback
    button_result = get_result(stt_button)

    ###nonUI_state.user_text_input_widget(session_state, value=session_state['query'])
    # ^ says no query key when i do that... which doesn't make much sense

    if button_result:
        if "GET_TEXT" in button_result:
            print("GET_TEXT in result")
            if button_result.get("GET_TEXT")["t"] != '' and button_result.get("GET_TEXT")["s"] != session_state['stt_session'] : # "s" for "session", "t" for "text"
                print("""result.get("GET_TEXT")["t"] != '' and result.get("GET_TEXT")["s"] != session_state['stt_session']""") 
                session_state['query'] = button_result.get("GET_TEXT")["t"]
                #nonUI_state.user_text_input_widget(tr, session_state)
                user_text_input_surrogate(tr, session_state)
                session_state['stt_session'] = button_result.get("GET_TEXT")["s"]

        if "GET_INTRM" in button_result:
            if button_result.get("GET_INTRM") != '':
                print("GET_INTRM != ''")
                user_text_input_surrogate(tr, session_state, value=session_state['query']+' '+button_result.get("GET_INTRM"))
                #st.text_area("**Your input**", value=session_state['query']+' '+button_result.get("GET_INTRM"))

        if "GET_ONREC" in button_result:
            if button_result.get("GET_ONREC") == 'start':
                #placeholder.image("recon.jpg")
                print("(GET_ONREC==start)")
                session_state['query'] = ''
            elif button_result.get("GET_ONREC") == 'running':
                #placeholder.image("recon.gif")
                print("(GET_ONREC==running)")
            elif button_result.get("GET_ONREC") == 'stop':
                #placeholder.image("recon.jpg")
                print("(GET_ONREC==stop)")
                #if session_state['query'] != '': # maybe irrelevant
                #    input = session_state['query']


def mic_button_monitor(tr, nonUI_state, stt_button, session_state):

    stt_button.js_on_event('button_click', get_mic_button_js()) # second argument is a callback
    button_result = get_result(stt_button)

    #nonUI_state.user_text_input_widget(session_state, value=session_state['query'])
    # ^ says no query key when i do that... which doesn't make much sense

    if button_result:
        if "GET_TEXT" in button_result:
            print("GET_TEXT in result")
            if button_result.get("GET_TEXT")["t"] != '' and button_result.get("GET_TEXT")["s"] != session_state['stt_session'] : # "s" for "session", "t" for "text"
                print("""result.get("GET_TEXT")["t"] != '' and result.get("GET_TEXT")["s"] != session_state['stt_session']""") 
                session_state['queried'] = button_result.get("GET_TEXT")["t"]
                nonUI_state.user_text_input_widget(tr, session_state)
                session_state['stt_session'] = button_result.get("GET_TEXT")["s"]

        if "GET_INTRM" in button_result:
            if button_result.get("GET_INTRM") != '':
                print("GET_INTRM != ''")
                nonUI_state.user_text_input_widget(tr, session_state, value=session_state['queried']+' '+button_result.get("GET_INTRM"))
                #st.text_area("**Your input**", value=session_state['query']+' '+button_result.get("GET_INTRM"))

        if "GET_ONREC" in button_result:
            if button_result.get("GET_ONREC") == 'start':
                #placeholder.image("recon.jpg")
                print("(GET_ONREC==start)")
                session_state['queried'] = ''
                #nonUI_state.user_text_input_widget(tr, session_state, value="") # EXPERIMENT
            elif button_result.get("GET_ONREC") == 'running':
                #placeholder.image("recon.gif")
                print("(GET_ONREC==running)")
            elif button_result.get("GET_ONREC") == 'stop':
                #placeholder.image("recon.jpg")
                print("(GET_ONREC==stop)")
                #if session_state['query'] != '': # maybe irrelevant
                #    input = session_state['query']
    #else:
    #    print("no button result")
    #    nonUI_state.user_text_input_widget(tr, session_state)
    #    #nonUI_state.user_text_input_widget(session_state)
    


def get_mic_button_js():
    code="""
    var value = "";
    var rand = 0;
    var recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'zh-CN';

    document.dispatchEvent(new CustomEvent("GET_ONREC", {detail: 'start'}));
    
    recognition.onspeechstart = function () {
        document.dispatchEvent(new CustomEvent("GET_ONREC", {detail: 'running'}));
    }
    recognition.onsoundend = function () {
        document.dispatchEvent(new CustomEvent("GET_ONREC", {detail: 'stop'}));
    }
    recognition.onresult = function (e) {
        var value2 = "";
        for (var i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) {
                value += e.results[i][0].transcript;
                rand = Math.random();
                
            } else {
                value2 += e.results[i][0].transcript;
            }
        }
        document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: {t:value, s:rand}}));
        document.dispatchEvent(new CustomEvent("GET_INTRM", {detail: value2}));

    }
    recognition.onerror = function(e) {
        document.dispatchEvent(new CustomEvent("GET_ONREC", {detail: 'stop'}));
    }
    recognition.start();
    """

    return CustomJS(code=code)