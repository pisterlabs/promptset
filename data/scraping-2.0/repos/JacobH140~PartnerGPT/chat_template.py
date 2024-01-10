import streamlit as st
from streamlit_chat import message
from utils import get_initial_message, get_chatgpt_response, get_chatgpt_response_stream_chunk, update_chat, stream_chat_completion, get_chatgpt_response_enforce_python_formatting
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
import string
from playsound import playsound
import base64
import chinese_nlp_utils as cnlp
import ast
import gsheet_utils as gs

# READ: ----------------------------------------------------------------
"""This file """



from streamlit_bokeh_events import streamlit_bokeh_events

from gtts import gTTS
from io import BytesIO
import openai
openai.api_key = api_key

def remove_text_inside_brackets(text, brackets="()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
                md,
                unsafe_allow_html=True,
            )

punctuation = r"""!?.;:。。()[]？。、；：-""" 
def has_punctuation(s):  
    return any([c in punctuation for c in s])

def remove_spaces_punctuation(input_string):
    # Remove spaces
    #input_string_no_space_punc = input_string.replace(" ", "")
    input_string_no_space_punc = input_string
    for c in input_string_no_space_punc:
        if c in punctuation:
            input_string_no_space_punc = input_string_no_space_punc.replace(c, "")
    
    # Remove punctuation
    input_string_no_space_punc = ' '.join(input_string_no_space_punc.split())
    return input_string_no_space_punc

class SessionNonUIState:
    def __init__(self, name):
        self.name = name
        self.custom = defaultdict(lambda: None) # an instance can have some 'custom' attributes that other instances don't
        self.end_message = "Conversation has ended... possibly unintentially."

        self.audio_playing = False

        self.simpl_or_trad = None # need to decide if this is a page-wide setting or not..

        self.next = None # this is a FUNCTION to update the state when next button is clicked... e.g., update the queue of new cards. Not always relevant (e.g., not relevant for translate)
        self.next_func_args = (None,)

        self.begin_disabled = False
        
        self.to_answer = defaultdict(list)
        self.to_create_prompt = None
        self.to_create_candidates = None

        self.initial_message_func = None
        self.initial_message_func_args = ("not set",)
        self.chatting_has_begun = False
        self.model = None
        self.messages = []
        self.prev_messages = []

        self.generated = []
        self.past = []
        self.reviewed = []

        self.submitted_ratings = defaultdict(lambda: None)
        self.submitted_rating =  defaultdict(lambda: None)
        self.administer_rating_form = False
        self.form_submit_button_clicked = False
        self.on_automatic_rerun = False # this is specific to the form thing... don't try to use it for anything else
        
        self.flag_generated_response_this_run = False # the response generation process updates UI in a streaming manner... but if it never gets called, we want to update UI manually

    def user_text_input_widget(self, tr, session_state_object, nonUI_state, value=None):
        if value is None:
            value = session_state_object[f"{nonUI_state.name}_query"]
            #tr.text_input("You: ", value=session_state_object['query'], key="query", placeholder='speak or type', label_visibility="collapsed", on_change=clear_text, disabled=self.administer_rating_form)
        #else:
        ended = self.custom['out_of_cards'] and nonUI_state.on_automatic_rerun 
        print("ended: ", ended)
        print("admin rating form:", self.administer_rating_form)
        print("or them together: ", self.administer_rating_form or ended)
        tr.text_input("You: ", key=f"{nonUI_state.name}_query", placeholder='begin / resume conversation', label_visibility="collapsed", on_change=clear_text, args=(nonUI_state,), disabled=ended or self.administer_rating_form)


    def stream_response(self, messages, real_time_audio=False):
        # real-time audio is experimental and will only work when running locally from mac. to get verbose mode, go to partnerGPT and copy over the comments used there
        with st.empty():
            # create variables to collect the stream of chunks
            collected_chunks = []
            collected_messages = []
            # iterate through the stream of events
            for chunk in stream_chat_completion(messages, self.model, temperature=0.5):
                collected_chunks.append(chunk)  # save the event response
                chunk_message = chunk['choices'][0]['delta']  # extract the message
                collected_messages.append(chunk_message)  # save the message
                response = ''.join([m.get('content', '') for m in collected_messages])
                prev_response_so_far = st.session_state['response_so_far']
                if len(response) > len(prev_response_so_far):
                   st.success(response)
                   to_sonify = response[len(st.session_state['sonified_so_far']):len(response)]
                   v = response[len(prev_response_so_far):len(response)]      
                   if has_punctuation(v) and real_time_audio: # only sonify clause-like structures, and  don't sonify parentheses contents 
                        split, langs = cnlp.split_text(to_sonify)
                        for s in range(len(split)):
                            if cnlp.remove_pinyin_tone_marked_ish(remove_spaces_punctuation(split[s])):
                                tts = gTTS(cnlp.remove_pinyin_tone_marked_ish(remove_spaces_punctuation(split[s])).strip(), lang=langs[s])
                                tts.save("sonify" + '.mp3')
                                playsound("sonify" + '.mp3')
                                st.session_state['sonified_so_far'] = response
                   st.session_state['response_so_far'] = response
                   st.session_state['currently_sonifying'] = to_sonify
                   st.success(response)
        
                st.success(response)
            if not real_time_audio and not self.audio_playing:
                pass
                ##tts = gTTS(remove_text_inside_brackets(response), lang='zh-cn')
                ##print(remove_text_inside_brackets(response))
                ##tts.save("response.mp3")
                ##autoplay_audio("response.mp3")
                ###st.success(response)
                ###update_UI_messages(self)

            if st.session_state['response_so_far'] == st.session_state['sonified_so_far']:
                st.session_state['response_so_far'] = ""
                st.session_state['sonified_so_far'] = ""

            return response

    def generate_bot_response_placeholder(self, query):
        messages = self.messages

        print("gbotp.self.custom['out_of_cards']: ", self.custom['out_of_cards'])
        if self.custom['out_of_cards'] == 1 and self.on_automatic_rerun:
            update_UI_messages(self)
            #st.warning("test")
            
            return messages

        messages = update_chat(messages, "user", query)
        self.past.append(query)
        update_UI_messages(self)
        response = self.stream_response(messages, real_time_audio=False) # call the model
        messages = update_chat(messages, "assistant", response)
        self.generated.append(response)
        self.reviewed.append("")
        self.flag_generated_response_this_run = True


        return messages
    


    def review_notif(self, str):
        self.generated.append("")
        self.past.append("")
        self.reviewed.append(str)

    def reset_submitted_ratings(self):
        self.submitted_ratings = defaultdict(lambda: None)
        assert(len(self.submitted_ratings) == 0)

    def rerun(self):
        self.on_automatic_rerun = True
        st.experimental_rerun()

def get_initial_message_placeholder():
    messages=[
            {"role": "system", "content": "system initial message placeholder"},
            {"role": "user", "content": "user initial message placeholder"},
            {"role": "assistant", "content": "assistant initial message placeholder"},
        ]
    return messages


    
 
def update_UI_messages(state_object, mode="stream"):
    for i in range(len(state_object.past)): 
        if state_object.past[i]:
            st.info(state_object.past[i])
        if mode=="stream" and i < len(state_object.generated) and state_object.generated[i]:
            st.success(state_object.generated[i])
        #elif state_object.generated[i]:
        #    st.success(state_object.generated[i])
        if mode=="stream" and i < len(state_object.generated) and state_object.reviewed[i]:
            st.warning(state_object.reviewed[i])
        #elif state_object.reviewed[i]:
        #    st.warning(state_object.reviewed[i])



def initialize_app(heading, subheading):
    st.title(heading)
    
    if 'state' not in st.session_state:
        st.session_state.state = SessionNonUIState()

    if not st.session_state.state.chatting_has_begun:
        model = st.selectbox("Select a model", ("gpt-3.5-turbo-16k", "gpt-4"))
        st.session_state.state.model = model
    
    return st.session_state.state

def expander_messages_widget(state_object):
    with st.sidebar.expander("Show Current Messages"):
        st.write(state_object.messages)
        

def rating_form(nonUI_state, to_answer=defaultdict(list), to_create_prompt=None): 
    def on_submit():
        nonUI_state.administer_rating_form = False
        nonUI_state.form_submit_button_clicked = True
        st.snow()
    print("RATING FORM FUNC CALLED, MESSAGES ARE: ", nonUI_state.messages)
    if to_create_prompt is not None and nonUI_state.administer_rating_form:
        nonUI_state.prev_messages.append({"role":"user", "content":f"""{to_create_prompt}"""})
        #print(nonUI_state.prev_messages)
        response = get_chatgpt_response_enforce_python_formatting(nonUI_state.prev_messages, response_on_fail = "[]", extra_prompt="What you are writing will be interpreted directly as Python code. So, make sure to give your response as a (possibly empty) LIST OF STRINGS, and nothing but a list of strings. No markdown.", start_temperature=0.5, model=nonUI_state.model)
        update_chat(nonUI_state.prev_messages, "assistant", response)
        nonUI_state.to_create_candidates = ast.literal_eval(response)
    #else:
    #    nonUI_state.to_create_candidates = []


    with st.form('Rating Form', clear_on_submit=False):
        ratings = defaultdict(lambda: None)
        answer_rows = []
        c = 0
        for word, id in zip(to_answer["text"], to_answer["ids"]):
            if c != 0:
                r = st.radio(f"**Review**: {word}", ("Not Reviewed", "Again", "Hard", "Good", "Easy"), horizontal=True)
            else:
                r = st.radio(f"**Review**: {word}", ("Again", "Hard", "Good", "Easy"), horizontal=True)
            print("WORD:", word)
            ratings[word] = r
            answer_rows.append([word, r, id])
            c += 1
        

        create_rows = []
        for word in nonUI_state.to_create_candidates:
            r = st.radio(f"**Create**: {word}?", ("Yes", "No"), horizontal=True)
            ratings[word] = r
            create_rows.append([word, r]) # will maybe later add context messages here, but not right now

        if nonUI_state.form_submit_button_clicked:
            nonUI_state.to_create_candidates = []
            nonUI_state.to_answer["text"] = []; nonUI_state.to_answer["ids"] = []

        print("RETURNING FROM RATING FORM FUNC, CREATE ROWS ARE", create_rows)
        print("RETURNING FROM RATING FORM FUNC, ANSWER ROWS ARE", answer_rows)
        return st.form_submit_button('Submit', on_click=on_submit), ratings, answer_rows, create_rows

def ratings_widget(state_object):
    with st.sidebar.expander("Show Review Ratings"):
        st.write(state_object.submitted_ratings)


def clear_text(nonUI_state): # should this have st.session_state as an argument?
    st.session_state[f"{nonUI_state.name}_queried"] = st.session_state[f"{nonUI_state.name}_query"]
    st.session_state[f"{nonUI_state.name}_query"] = ""

def UI_controls(nonUI_state): 
    st.divider()
    mic, user, next_button = st.columns([2,30,4])
    with mic:
        st.button("🎙️", key="mic_button", disabled=True)
        #if 'stt_session' not in st.session_state:
        #    st.session_state['stt_session'] = 0 # init
        #stt_button = stt.mic_button()
    with user:
        if f"{nonUI_state.name}_query" not in st.session_state:
            st.session_state[f"{nonUI_state.name}_query"] = ''
        tr = st.empty()
        nonUI_state.user_text_input_widget(tr, st.session_state, nonUI_state)
        #stt.mic_button_monitor(tr, nonUI_state, stt_button, st.session_state) 
        
    with next_button:
        ended = nonUI_state.custom['out_of_cards'] and nonUI_state.custom['out_of_cards'] and nonUI_state.custom['out_of_cards'] == 1 and nonUI_state.on_automatic_rerun 
        st.button("Next", key="next_button", on_click=on_proceed_button_click, args=(nonUI_state,), disabled=ended or nonUI_state.administer_rating_form)



def on_proceed_button_click(nonUI_state):
    print("on_proceed_button_click is running")
    print("nonUI_state.custom['out_of_cards']: ", nonUI_state.custom['out_of_cards'])
    flag = False
    if nonUI_state.custom['out_of_cards'] and nonUI_state.custom['out_of_cards'] > 1 and not nonUI_state.administer_rating_form:
        st.warning(nonUI_state.end_message)
        flag = True
    #nonUI_state.messages = nonUI_state.initial_message_func(nonUI_state.vocab_words_testing_temp.pop(), nonUI_state.aux_words_testing_temp)
    nonUI_state.prev_messages = nonUI_state.messages
    nonUI_state.messages = nonUI_state.initial_message_func(*nonUI_state.initial_message_func_args)
    



    #print(nonUI_state.messages)
    if nonUI_state.chatting_has_begun:
        nonUI_state.administer_rating_form = True
        clear_text(nonUI_state)



    if flag:
        return
    
    if f"{nonUI_state.name}_query" in st.session_state and not nonUI_state.administer_rating_form: # first part ensures this doesn't run when 'begin' is pressed, second part ensures 'next' behavior doesn't run while the rating form is being administered

        st.session_state[f"{nonUI_state.name}_queried"] = f'{nonUI_state.name} — **continue !**'
        nonUI_state.messages = nonUI_state.generate_bot_response_placeholder(query=st.session_state[f"{nonUI_state.name}_query"])
        clear_text(nonUI_state)
    elif not nonUI_state.administer_rating_form:
        print(nonUI_state.messages)
        nonUI_state.messages = nonUI_state.generate_bot_response_placeholder(query=f'{nonUI_state.name} — **begin !**')
        #st.markdown("<details> <summary>Details</summary> Something small enough to escape casual notice. </details>", unsafe_allow_html=True)
    
    nonUI_state.chatting_has_begun = True


def chat(nonUI_state):
    if f"{nonUI_state.name}_queried" not in st.session_state:
        st.session_state[f"{nonUI_state.name}_queried"] = ""

    if 'response_so_far' not in st.session_state:
        st.session_state['response_so_far'] = ""

    if 'sonified_so_far' not in st.session_state:
        st.session_state['sonified_so_far'] = ""

     

    if not nonUI_state.chatting_has_begun:
        st.button("Begin", key="begin_button", on_click=on_proceed_button_click, args=(nonUI_state,), disabled=nonUI_state.begin_disabled)


    if f"{nonUI_state.name}_query" in st.session_state and st.session_state[f"{nonUI_state.name}_queried"]:
        nonUI_state.messages = nonUI_state.generate_bot_response_placeholder(st.session_state[f"{nonUI_state.name}_queried"])

    if nonUI_state.generated: 
        if nonUI_state.on_automatic_rerun: # first part ensures this doesn't run when 'begin' is pressed, second part ensures 'next' behavior doesn't run while the rating form is being administered
            st.session_state[f"{nonUI_state.name}_queried"] = '**continue !**'
            nonUI_state.review_notif(f"**Submitted ratings!**")
            nonUI_state.messages = nonUI_state.generate_bot_response_placeholder(query=st.session_state[f"{nonUI_state.name}_queried"])
    
    #if nonUI_state.generated and not nonUI_state.flag_generated_response_this_run:
    #    update_UI_messages(nonUI_state, mode="stream")

    if nonUI_state.administer_rating_form or nonUI_state.form_submit_button_clicked: #nonUI_state.on_automatic_rerun:
        submitted_form, nonUI_state.submitted_rating, answer_rows, create_rows = rating_form(nonUI_state, nonUI_state.to_answer, nonUI_state.to_create_prompt) # only want to call rating form when administer_rating_form is True, but only care about the results when it's false...?


        if nonUI_state.form_submit_button_clicked and not nonUI_state.on_automatic_rerun: # note that at all times AT MOST one of form_submit_button_clicked, administer_rating_form, and on_automatic_rerun can be True
            print("CREATE ROWS BEFORE THINGS ARE FILTERED OUT", create_rows)
            create_rows = [row for row in create_rows if row[1] == "Yes"]
            create_rows = [[row[0], nonUI_state.name] for row in create_rows] # subsequent filtering
            answer_rows = [row for row in answer_rows if row[1] != "Not Reviewed"]
            
            print("SHOULD BE CREATING ROWS", create_rows)
            gs.add_rows_to_gsheet(create_rows, "To Create")
            gs.add_rows_to_gsheet(answer_rows, "Answered Cards Cache")
            print("updated 'to create' and 'review cards cache' in gsheet")
            if nonUI_state.next is not None:
                nonUI_state.next(*nonUI_state.next_func_args)
            nonUI_state.form_submit_button_clicked = False
            nonUI_state.rerun()

            
    if nonUI_state.on_automatic_rerun: # this is when the submitted ratings are the real ones
        nonUI_state.submitted_ratings = nonUI_state.submitted_ratings | nonUI_state.submitted_rating # dictionary merge
    
    if nonUI_state.custom['out_of_cards'] and nonUI_state.on_automatic_rerun:
        st.warning(nonUI_state.end_message)

    
    if nonUI_state.chatting_has_begun:
        UI_controls(nonUI_state)

                

    
    if nonUI_state.generated:
        expander_messages_widget(nonUI_state)
        ratings_widget(nonUI_state)
    


    nonUI_state.flag_generated_response_this_run = False # reset this flag

if __name__ == '__main__':
    nonUI_state = initialize_app("header", "subheader")
    chat(nonUI_state)
    if nonUI_state.on_automatic_rerun:
        nonUI_state.on_automatic_rerun = False
    

#https://stackoverflow.com/questions/52718897/minimal-shortest-html-for-clickable-show-hide-text-or-spoiler