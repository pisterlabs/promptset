import utils
import streamlit as st
from prop_agent import agent
from langchain.callbacks import StreamlitCallbackHandler
from streaming import StreamHandler

st.set_page_config(
    page_title="PropGPT",
    page_icon='🏠',
    layout='wide'
)
st.header('PropGPT', divider= 'rainbow')
st.caption(':blue[_beta_]')

@utils.enable_chat_history
# def main():
#     user_query = st.chat_input(placeholder= "Enter your Query here:")
#     if user_query:
#         utils.display_msg(user_query, 'user')
#         with st.chat_message("PropLlama"):
#             st_cb = StreamlitCallbackHandler(st.container())
#             response = agent.run(user_query, callbacks=[st_cb])
#             st.session_state.messages.append({"role": "PropLlama", "content": response})
#             st.write(response)

def main():
        user_query = st.chat_input(placeholder="Ask your Query")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("PropGPT"):
                st_cb = StreamlitCallbackHandler(st.container())
                response = agent.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "PropGPT", "content": response})
                st.write(response)

if __name__ == "__main__":
    main()
