import streamlit as st
from typing import Dict, Text, Any
import json

from langchain.schema import (
    HumanMessage,
    AIMessage
)
from peloton import PelotonAPI
from agent import PeloAgent


@st.cache_data()
def load_goals() -> Dict[Text, Any]:
    """Loads the user fitness goals defined in goals.json to populate the goals dropdown."""
    try:
        return json.load(open('goals.json', 'r'))
    except FileNotFoundError:
        return {}


def reset():
    """Resets the session state and cache."""
    persona = goal_map[goal]["goal"]
    st.session_state["agent"] = PeloAgent(persona)
    
    st.cache_data.clear()


# Setup the sidebar.
with st.sidebar:
    st.title("Peloton GPT Personal Trainer")
    goal_map = load_goals()
    if len(goal_map) == 0:
        st.markdown("Define a persona to customize your experience.")
        goal = ""
    else:
        goal = st.selectbox(
            label="Choose a goal",
            options=list(goal_map.keys()),
            on_change=reset
        )
        
    get_workout = st.button("Generate Workout")


if "agent" not in st.session_state:
    # Load the selected persona.
    if goal != "":
        persona = goal_map[goal]["goal"]
    else:
        persona = ""
    st.session_state["agent"] = PeloAgent(persona)


if "pelo_interface" not in st.session_state:
    pelo = PelotonAPI()
    pelo_auth = pelo.authenticate()
    user_id = pelo_auth.json()['user_id']
    st.session_state["pelo_interface"] = pelo
    st.session_state["pelo_user_id"] = user_id


# Display the chat.
for msg in st.session_state["agent"].chat_history:
    content = msg.content
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f'*:grey["{content}"]*')
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(content)


user_input = st.chat_input()

if get_workout or user_input:
    if not user_input:
        user_input = "What is my recommended workout today?"

    # Add the user input to the chat.
    with st.chat_message("user"):
        st.markdown(f'*:grey["{user_input}"]*')
        output = st.session_state["agent"].invoke(user_input)
    
    # Generate the workout.
    with st.spinner("Generating your workout..."):
        with st.empty():
            with st.chat_message("assistant"):
                st.markdown(output)
