import streamlit as st
from typing import List
from memory.memory_utils import get_summary_and_generate_message, search_full_memory_and_rerank
import openai

system_message = {"role": "system", "content": "You are an assistant that provides information based on context. You are to respond to questions based on the context provided and only the context. Don't add information not found in the context. Be concise and friendly.\nContext: {}"}

def view_memory():
    if(st.session_state['view_user'] == "" or st.session_state['view_session'] == ""):
         st.header("Add a user and a session to get started!", divider=True)
    
    else:
        user = st.session_state['view_user'] 
        session = st.session_state['view_session']

        if "view_messages" not in st.session_state:
            st.session_state["view_messages"] = []
            # Generate first message from summary
            summary = get_summary_and_generate_message(user=user, session=session)
            if summary != None:
                st.session_state["view_messages"].append({"role": "assistant", "content": "Here is a summary of this memory:\n\n" + summary + "\n\nWhat would you like to know?"})
            else:
                st.session_state["view_messages"].append({"role": "assistant", "content": "What would you like to know?"})

        st.header("View Memory", divider=True)

        for msg in st.session_state.view_messages:
            if(msg["role"] != "system"):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        
        if prompt := st.chat_input():
            
            context_retrieved = search_full_memory_and_rerank(query=prompt,user=user, session=session)
            if len(context_retrieved) > 0:
                context = "\n".join(f"{result['text']}" for result in context_retrieved)
            else: 
                context_retrieved = []
                context = "No context available"
            if st.session_state['debug']:
                st.text_area("Context",context)
            system_message_temp = system_message
            system_message_temp['content'] = system_message_temp['content'].format(context)

            st.session_state.view_messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            # Should also create user and sessions for this?
            history_to_send = [system_message_temp] + st.session_state.view_messages
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in openai.ChatCompletion.create(model="gpt-4", messages=history_to_send, stream=True):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.view_messages.append({"role": "assistant", "content": full_response})
    
