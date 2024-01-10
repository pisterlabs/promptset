import streamlit as st
from openai import OpenAI
import time

st.set_page_config(page_title="Chat GPT", page_icon=":flag-pk:", layout="centered")
# st.title("")



container = st.container()

# Create two containers with a border using markdown
tab1, tab2 = st.columns(2)
st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #000000;
            color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)




# Initialize the OpenAI API client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# set the default model
if "model" not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo-1106"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    # clear the chat history
with st.sidebar:
    st.markdown("""
        <style>
            [data-testid=stSidebar] {
                background-color: #000000;
                color: #ffffff;
            }
        </style>
        """, unsafe_allow_html=True)
    if st.button("Clear conversation", key="clear", type="primary"):
        st.session_state.messages = []


            

if prompt := st.chat_input("Message ChatGPT..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        # message_box = f"<div style='background-color:#f2f2f2; padding:10px; border-radius:10px; margin-bottom:10px; height:200px; overflow-y:scroll; color:black;'>{full_response}</div>"
        # st.markdown(message_box, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


   
else:
    with container:
        st.subheader("""
              HI :wave: , This is ChatGPT Clone, 
              I am Aqeel Shahzad a WMD student at PIAIC Institue of Science and Technology.
              I know JAMstack developement and learning generative AI applications in python . """)
        st.markdown("This is my [GitHub](https://github.com/aqeel-spec) follow me!")
        
    with container:
        html_code_start = f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;  padding: 20px; border-radius: 10px; margin-bottom: 20px; margin-top: 20px; ">
                <div class="flex-item-image" style="text-align: center; justify-content: center , padding: 20px,border:2px solid , border-radius: 50% ,border-radius: 50%; margin-bottom: 10px " >
                    <img src="https://seeklogo.com/images/C/chatgpt-logo-02AFA704B5-seeklogo.com.png" width=30px height=30px alt="GPT Logo">
                </div>
                <div class="flex-item-text" style="text-align: center; justify-content: center;">
                    <h3>How can I help you today?</h3>
                </div>
            </div>
        """
        st.markdown(html_code_start, unsafe_allow_html=True)
    with tab1:
        html_code = f"""
            <div  style='border:1px solid #e6e6e6; padding: 10px; border-radius: 10px;cursor: pointer'>
                <div style="color: black; font-weight: bold ">Compare storytelling techniques</div>
                <p style="color:"#40414F"; opacity:0.5% ">in novels and in films</p>
            </div>
            """
        st.markdown(html_code, unsafe_allow_html=True)
    with tab2:
        html_code = f"""
            <div  style='border:1px solid #e6e6e6; padding: 10px; border-radius: 10px; cursor: pointer'>
                <div style="color: black;fontsize: 14px; font-weight: bold;  ">Design a database schema</div>
                <p style="color:"#40414F"; opacity:0.5% ">for an online merch store</p>
            </div>
            """
        st.markdown(html_code, unsafe_allow_html=True)