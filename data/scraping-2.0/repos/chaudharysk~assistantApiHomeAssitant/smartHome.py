import openai
import streamlit as st
import time
import os
import json
import yfinance as yf
from dotenv import load_dotenv
import Tuya
import credential


def handle_function(run):
    print("Inside ------>handle_function")
    tools_to_call = run.required_action.submit_tool_outputs.tool_calls
    print("tools_to_call:")
    print(tools_to_call)
    tools_output_array = []
    for each_tool in tools_to_call:
        tool_call_id = each_tool.id
        function_name = each_tool.function.name
        function_arg = each_tool.function.arguments
        print("Tool ID:" + tool_call_id)
        print("Function to Call:" + function_name )
        print("Parameters to use:" + function_arg)

        if (function_name == 'switch_light'):
            arguments_str = each_tool.function.arguments
            arguments_dict = json.loads(arguments_str)
            state = arguments_dict['state']
            st.sidebar.write('Sending Command...', state)
            output = Tuya.switch_light(state)
            tools_output_array.append({"tool_call_id": tool_call_id, "output": output})

        elif (function_name == 'get_light_status'): 
            st.sidebar.write('Getting light status')
            st.sidebar.write('Sending Command...')
            output = Tuya.get_light_status()
            print(output)
            tools_output_array.append({"tool_call_id": tool_call_id, "output": str(output)})

        elif (function_name == 'set_brightness'):
            arguments_str = each_tool.function.arguments
            arguments_dict = json.loads(arguments_str)
            brightness = arguments_dict['brightness']
            st.sidebar.write('Setting brightness to ', brightness)
            output = Tuya.set_brightness(brightness)
            tools_output_array.append({"tool_call_id": tool_call_id, "output": output})

        elif (function_name == 'set_tempearture'):
            arguments_str = each_tool.function.arguments
            arguments_dict = json.loads(arguments_str)
            temp = arguments_dict['temp']
            st.sidebar.write('Setting temperature to ', temp)
            output = Tuya.set_temperature(temp)
            tools_output_array.append({"tool_call_id": tool_call_id, "output": output})

    client.beta.threads.runs.submit_tool_outputs(
        thread_id = st.session_state.thread_id,
        run_id = run.id,
        tool_outputs=tools_output_array
    )    


# Set your OpenAI Assistant ID here
assistant_id = credential.OpenAIassistant_id

# Initialize the OpenAI client (ensure to set your API key in the sidebar within the app)
client = openai

# Initialize session state variables for file IDs and chat control
if "file_id_list" not in st.session_state:
    st.session_state.file_id_list = []

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Set up the Streamlit page with a title and icon
st.set_page_config(page_title="GPT-3 Powered Smart Home", page_icon=":bulb:", layout="wide")
st.header(":robot_face: GPT-3 Powered Smart Home:bulb:")

#Get the OPENAI API Key 
openai_api_key_env = credential.OpenAIapi
openai_api_key = st.sidebar.text_input(
    'OpenAI API Key', placeholder='sk-', value=openai_api_key_env, type="password")
url = "https://platform.openai.com/account/api-keys"
st.sidebar.markdown("Get an Open AI Access Key [here](%s). " % url)
if openai_api_key:
    openai.api_key = openai_api_key

# Button to start the chat session
if st.sidebar.button("Start Chat"):
    st.session_state.start_chat = True
    # Create a thread once and store its ID in session state
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id
    st.write("thread id: ", thread.id)

# Define the function to process messages with citations
def process_message_with_citations(message):
    message_content = message.content[0].text.value
    return message_content

# Only show the chat interface if the chat has been started
if st.session_state.start_chat:
   # st.write(getStockPrice('AAPL'))
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for the user
    if prompt := st.chat_input("How can I help you?"):
        # Add user message to the state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add the user's message to the existing thread
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=prompt
        )

        # Create a run with additional instructions
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=credential.OpenAIassistant_id,
            instructions="Help in smart bulb decisions",
        )

        # Poll for the run to complete and retrieve the assistant's messages
        while run.status not in ["completed", "failed"]:
            st.sidebar.write(run.status)
            if run.status == "requires_action":
                handle_function(run)
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id
            )
        st.sidebar.write(run.status)

        # Retrieve messages added by the assistant
        messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread_id
        )

        # Process and display assistant messages
        assistant_messages_for_run = [
            message for message in messages 
            if message.run_id == run.id and message.role == "assistant"
        ]
        for message in assistant_messages_for_run:
            full_response = process_message_with_citations(message)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant"):
                st.markdown(full_response, unsafe_allow_html=True)