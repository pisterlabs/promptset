import openai
import streamlit as st
from streamlit_pills import pills
import Constants

openai.api_key = Constants.API_KEY

st.sidebar.markdown("<h1 style='color: grey;'>BYUH Faculty of Math and Computing</h1>", unsafe_allow_html=True)

st.subheader("AI Assistant: Ask Me Anything")

# Storing the chat in a session
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# Display chat history
for past_interaction in st.session_state['past']:
    if past_interaction['role'] == 'user':
        st.markdown(f'<div style="border:2px solid coral; padding:10px; margin:5px; border-radius: 15px;">You: {past_interaction["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="border:2px solid LightBlue; padding:10px; margin:5px; border-radius: 15px;">AI: {past_interaction["content"]}</div>', unsafe_allow_html=True)

res_box = st.empty()


selected = pills("", ["Streaming", "NO Streaming"], ["😎", "😊"])

user_input = st.text_input("You", placeholder="Ask me a question here...", key="input")

if st.button("Submit"):
    st.markdown("____")

    # Manage history
    history = [{"role": "system", "content": "You are very knowledgeable in sports, dating life, tips, and fun facts. Answer the following questions in a concise way, please write at least 4 sentences."}]
    history.extend(st.session_state['past'])
    history.append({"role": "user", "content": user_input})

    if selected == "Streaming":
        report = []
        for resp in openai.chat.completions.create(model='gpt-4',
                                                   messages=history,
                                                   n=1,
                                                   max_tokens=1024,
                                                   temperature=0.5,
                                                   stream=True):
            content = resp.choices[0].delta.content
            if content is not None:
                report.append(content)
                current_output = "".join(report).strip()
                res_box.markdown(f'<div style="border:2px solid lightgreen; padding:10px; margin:5px; border-radius: 15px;"><b>Current Output: </b>{current_output}</div>', unsafe_allow_html=True)
                st.session_state['generated'].append(current_output)

    else:
        completions = openai.chat.completions.create(model='gpt-3.5-turbo-1106',
                                                     messages=history,
                                                     n=1,
                                                     max_tokens=1024,
                                                     temperature=0.5,
                                                     stream=False)

        result = completions.choices[0].message.content
        res_box.markdown(f'<div style="border:2px solid lightgreen; padding:10px; margin:5px; border-radius: 15px;"><b>Current Output: </b>{result}</div>', unsafe_allow_html=True)
        st.session_state['generated'].append(result)

    st.session_state['past'].append({"role": "user", "content": user_input})
    st.session_state['past'].append({"role": "assistant", "content": st.session_state['generated'][-1]})