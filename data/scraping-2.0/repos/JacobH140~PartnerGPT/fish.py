import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS

from streamlit_bokeh_events import streamlit_bokeh_events

import time
from gtts import gTTS
from io import BytesIO
import openai
import stt
import UI

TODO= None




if 'state' not in st.session_state:
    st.session_state['state'] = UI.SessionNonUIState()

nonUI_state = st.session_state.state # state error is throw iff this is placed on line 20

mic, user, next_button = st.columns([2,30,4])


#nonUI_state = st.session_state.state # state error is throw iff this is placed on line 20
with mic:
    if 'stt_session' not in st.session_state:
        st.session_state['stt_session'] = 0 # init
    stt_button = stt.mic_button()

with user:
    if 'query' not in st.session_state:
        st.session_state['query'] = ''
    tr = st.empty()
    #nonUI_state.user_text_input_widget(tr, st.session_state)
    stt.user_text_input_surrogate(tr, st.session_state)
    stt.mic_button_monitor_surrogate(tr, stt_button, st.session_state) 

with next_button:
    st.button("Next", key="next_button", disabled=True) 

