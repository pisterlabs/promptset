
'''
author: Roberto Scalas 
date:   2023-07-17 10:34:58.351165
'''
from scripts.utils import *
import openai
import streamlit as st
st.set_page_config(layout="wide")

from streamlit_ace import st_ace

DEFAULT_CODE = '''
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
options_for_ticker = [
    'eurusd=x',
    'gbpusd=x',
    'usdjpy=x',
    'usdchf=x',
    'TSLA',
    'AAPL',
    'MSFT',
    'AMZN',
    'GOOG',
    'VUSA.L',

]
import datetime
ticker = st.sidebar.selectbox('Select ticker', options_for_ticker)
interval = st.sidebar.selectbox('Select interval', ['1d', '1h', '30m', '15m', '5m', '1m'])
start_date = st.sidebar.date_input('Start date', value=datetime.date(2021, 1, 1))
end_date = st.sidebar.date_input('End date', value=datetime.date(2021, 7, 1))
def function():
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    # add plotly graph
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],

                    close=data['Close'], name = 'market data'))
    fig.update_layout(
        title='{} Stock Chart'.format(ticker),
        yaxis_title='Stock Price (USD per Shares)')
    st.plotly_chart(fig, use_container_width=True)
    return data


'''

conversations_box = st.empty()

code = st_ace(value=DEFAULT_CODE, language='python', theme='monokai', keybinding='vscode', font_size=12, tab_size=4, show_gutter=True, show_print_margin=True, wrap=True, auto_update=True, readonly=False, key=None)

openai.api_key = st.sidebar.text_input('OpenAI API Key', value='', max_chars=None, key=None, type='password', help=None)

def chat_with_gpt(prompt):
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    answer = chat_completion.choices[0].message['content']
    with st.expander('Answer'):
        st.markdown(answer)
    return answer

# create a chat with gpt
import datetime
from datab import Database_Questions
db = Database_Questions()
# get the conversations
conversations = db.select()

if len(conversations) > 0:
    # get unique conversations ids 
    conversations_ids = list(set([conversation[0] for conversation in conversations]))
    with st.sidebar.form(key='my_form_for_conversation'):
        conversation_id = st.selectbox('Select conversation', conversations_ids)
        c1,c2 = st.columns(2)
        submit_button = c1.form_submit_button(label='Submit', help=None, on_click=None, args=None, kwargs=None, use_container_width=True)
        new_conversation = c2.form_submit_button(label='New Chat', help=None, on_click=None, args=None, kwargs=None, use_container_width=True)
        delete_conversation = st.form_submit_button(label='Delete Chat', help=None, on_click=None, args=None, kwargs=None, use_container_width=True)
    
    if submit_button:
        # get all the questions and answers from the conversation
        conversations_id = str(conversation_id)

        # new conversation button
    if new_conversation:
        conversation_id = str(datetime.datetime.now())
        # reload the page
        # save a empty conversation
        db.insert(conversation_id, '', '', str(datetime.datetime.now()))
        st.experimental_rerun()

    # create a delete button
    if delete_conversation:
        db.delete_single_conversation_form_id(conversation_id)
        #st.experimental_rerun()
else:
    conversation_id = 'First conversation'

with conversations_box.expander(f'Conversation {conversation_id}', expanded=True):
    questions_answers = db.get_from_conversation_id(conversation_id)

    # each row contains a question and an answer
    if len(questions_answers) > 0:
        for row in questions_answers:
            # if the question is not empty
            if row[1] != '':
                # use the chat module
                with st.chat_message('User'):
                    st.write(row[1])
                with st.chat_message('assistant'):
                    st.write(row[2])
    else:
        st.write('No questions and answers for this conversation')

def on_click():
    # get the answer and save it
    question = st.session_state.question
    answer = chat_with_gpt(question)
    # get conversation id
    db.insert(conversation_id, question, answer, str(datetime.datetime.now()))

question = st.chat_input(placeholder='Type a question...', on_submit=on_click, key = 'question')


# Store the code as a string
stored_code = code

# Later, execute the stored code to define the function
exec(stored_code)

# Now, you can call the dynamically defined function
result = function()
st.write(result)  # Output: ciao
