import cohere 
import streamlit as st

co = cohere.Client('ECmZASmuWPzS4Np9LEgveRZkD9RZ3GTaD3li6rmJ') # This is your trial API key

import streamlit as st

st.title("SynthBot")
preamble_prompt = "Your name is synth bot. You're virtual asistant for JuriSynth an ai assistant for Syntheia corp"
docs = [
    {
        "name": "Syntheia Workspace",
        "desc": "Assistant for syntheia",
    }
]
def cohereChat(prompt):
        lim_response = co.chat(
        # message="hi"
        model = "command",
        message = prompt,
        preamble_override = preamble_prompt,
        documents=docs,
        )
        return lim_response

# Initialize chat history
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    init_state()
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        lim_response = cohereChat(prompt)
        response = f"{lim_response.text}"

        #display assist controller
        with st.chat_message("assistan"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
#response check
    
    
    
if __name__ == "__main__":
    main()