import streamlit as st
import langchain as lc

from typing import Callable
from utils import *

#####################################################
# This file contains everything reusable in the app #
#####################################################

def show_past_conversations():
    conversations = get_conversation_list()
    if len(conversations) <= 0:
        st.write("No past conversations")
    current_conversation_title = st.selectbox(
        "Conversations", 
        conversations, 
        on_change=del_old_chat, 
        index=0 if ("Conversation" not in st.session_state) or (not st.session_state["Conversation"].started) else conversations.index(st.session_state["Conversation"].conversation_name),
        help="Select a previous conversation to review. You can also start a new conversation by selecting 'New conversation'"
    )
    return current_conversation_title

def show_usage_stats():
    monthly_limit = st.number_input("Monthly limit ($)", value=15.0, min_value=1.0, max_value=120.0, step=1.0, format="%.2f", help="The monthly limit for the OpenAI API")
    day_total = st.session_state["UsageLogger"].day_total()
    month_total = st.session_state["UsageLogger"].month_total()
    prev_cost = st.session_state["UsageLogger"].prev_cost
    avg_cost = st.session_state["UsageLogger"].avg_query_cost()
    st.metric("Usage cost today", 
              "${:.6f} ({:.1f}%)".format(day_total, day_total/monthly_limit*100),
              "{:.6f} ({:.1f}%)".format(prev_cost, prev_cost/monthly_limit*100) if prev_cost > 0 else None,
              help="The total cost for the current day, and the percentage of the monthly limit used today"
              )
    st.metric("Usage cost this month", 
              "${:.6f} ({:.1f}%)".format(month_total, month_total/monthly_limit*100), 
              #"{:.6f} ({:.1f}%)".format(prev_cost, prev_cost/monthly_limit*100) if prev_cost > 0 else None,
              help="The total cost for the current month, and the percentage of the monthly limit currently used")
    st.metric("Average query cost", "${:.6f}".format(avg_cost), 
              "{:.6f}".format(prev_cost-avg_cost) if prev_cost > 0 else None,
              help="The average cost per prompt over all time")

def chat(create_model: Callable[[None], lc.chains.base.Chain]):
    ## Print previous messages
    if st.session_state["Conversation"].messages:
        for i in st.session_state["Conversation"].messages:
            st.chat_message(i['role']).write(i['content'])
    ## Get new message and response
    if prompt := st.chat_input():
        if "ChatBot" not in st.session_state:   # Create chat model. We don't want to create it before the user has written the first input.
            st.session_state["ChatBot"] = create_model()
        st.chat_message("User").write(prompt)
        st.session_state["Conversation"].append({'role': 'User', 'content': prompt})
        with st.spinner('Waiting for response...'):
            with lc.callbacks.get_openai_callback() as cb:
                response = st.session_state["ChatBot"].run(prompt)
        st.chat_message("Assistant").write(response)
        st.session_state["Conversation"].append({'role': 'Assistant', 'content': response})
        st.session_state["UsageLogger"].append(cb)
        st.experimental_rerun()   # To update metrics and widgets just in time.