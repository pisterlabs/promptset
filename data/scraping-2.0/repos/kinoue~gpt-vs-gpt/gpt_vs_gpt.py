import json
import streamlit as st
from streamlit_chat import message
from openai_client import ask_opinion
from db_helper import add_dialogue

st.header("GPT vs GPT")
st.write('''
    Ask GPT to discuss any topics. You specifiy the topic of the discussion, and
    GPT will generate a discussion between the two opposing views on the topic.
''')

st.write('''
    You can also brows the topics people have put [here](history).
''')

st.sidebar.success("Select a page above")

if 'statements' not in st.session_state:
    st.session_state['statements'] = []

control_area = st.empty()
dialogue_area = st.empty()

def ask_gpt():
    if not topic:
        st.error("Please provide the topic first.")
    else:
        st.session_state.statements = []
        _statements = st.session_state.statements
        for turn in range(num_turns):
            if len(_statements) == 0:
                _statements.append(ask_opinion(topic, stance, temperature))
                _statements.append(ask_opinion(topic, stance, temperature,  _statements[-1]))
            else:
                _statements.append(ask_opinion(topic, stance, temperature,  _statements[-1], _statements[-2]))
                _statements.append(ask_opinion(topic, stance, temperature,  _statements[-1], _statements[-2]))

    add_dialogue(
            topic=topic,
            stance=stance,
            num_turns=num_turns,
            temperature=temperature,
            statements=_statements
    )

with control_area.container():
    topic = st.text_input(
        label='Topic (e.g. "birth control", "the best basketball player in the history", "who is the strongest Avenger?"):',
        key="topic"
    )
    stance = st.select_slider(
        label='Stance: ',
        options=["Moderate", "Constructive", "Radical"],
        value="Constructive"
    )
    num_turns = st.select_slider(
        label='Number of Turns: ',
        options=[1, 2, 3, 4, 5],
        value=3
    )
    temperature = st.select_slider(
        label='Temperature: ',
        options=[o / 10 for o in range(0, 11, 1)], # range takes only int
        value=0.5
    )
    st.button("Ask GPT", on_click=ask_gpt)

with dialogue_area.container():
    if st.session_state.statements:
        for i, statement in enumerate(st.session_state.statements):
            print(f"{i}: {statement}")
            message(
                statement,
                is_user=(i % 2 == 1),
                key=str(i) + '_user',
                avatar_style='bottts',
                seed=(i % 2)
            )