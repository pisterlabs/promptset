import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from arduino_reader import ArduinoReader

# AI imports
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

try:
    arduino_reader = ArduinoReader(port='/dev/ttyUSB0', baud_rate=9600)
except:
    arduino_reader = None
def init():
    st.set_page_config(
        page_title="HealthMate Chat ⚕️",
        page_icon="🏥",
        initial_sidebar_state="expanded",
    )
    load_dotenv()

    # Test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.stop()

    # Initialize session state
    if "health_data" not in st.session_state:
        st.session_state["health_data"] = {
            "Heart Rate": 80.0,
            "Temperature": 98.6,
        }

def on_button_click():
    pass

def ai_chat():
    st.title("👨‍⚕️ HealthMate Chat")
    st.markdown("Welcome to HealthMate Chat, where you can get personalized health advice!")
    st.markdown('---')

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are the best doctor; you can answer any question asked to you. You also like to give advice and motivate people to live a healthy life and you also have a good sense of humor. you also like to use emojis in your conversation."),
        ]

    user_input = st.chat_input("Your message: ", key="user_input")

    if user_input:
        # Add health data to user input
        user_input += "\nHere are my vitals:\n"
        vitals = st.session_state["health_data"]
        for key in vitals:
            user_input += f"{key}: {vitals[key]}\n"

        user_input += "\nuse this data to answer my question\n"

        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking..."):
            try:
                response = chat(st.session_state.messages)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"Error generating AI response: {str(e)}")

    with st.sidebar:
        st.title("📊 Health Data Entry")
        st.write("Choose how to enter your health data.")
        health_data_method = st.selectbox("How do you want to enter your health data?", ("Enter Manually", "Read from Arduino"))

        try:
            if health_data_method == "Enter Manually":
                # Get all the health data; use keys from dict
                for key in st.session_state["health_data"]:
                    # Take input for each key
                    st.session_state["health_data"][key] = st.number_input(label=key, value=float(st.session_state["health_data"][key]))

            elif health_data_method == "Read from Arduino":
                st.subheader("Arduino Data Reading")
                duration_sec = st.number_input("Duration of data reading (seconds):", value=10)
                if st.button("Start Reading"):
                    with st.spinner("Reading data from Arduino..."):
                        # Read data from the Arduino for 10 seconds
                        if arduino_reader is None:
                            st.error("Arduino not connected!")
                            st.stop()
                        avg_heart_rate, avg_temperature = arduino_reader.read_data_avg(duration_sec)
                        # Update the health data
                        st.session_state["health_data"]["Heart Rate"] = avg_heart_rate
                        st.session_state["health_data"]["Temperature"] = avg_temperature
                        st.success("Data read successfully!")

                    # Show the data
                    st.subheader("Data Read from Arduino")
                    st.write(f"Average Heart Rate: {avg_heart_rate}")
                    st.write(f"Average Temperature: {avg_temperature}")

            # Add field
            st.markdown('---')
            st.title("➕ Add Field")
            st.write("Add a new health data field.")
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name").capitalize()
            with col2:
                value = st.text_input("Value")
            if st.button("Add"):
                # Get keys from dict
                names = st.session_state["health_data"].keys()
                if name in names:
                    st.error("Field already exists")
                else:
                    st.session_state["health_data"][name] = value
                    st.rerun()

            # Remove field
            st.markdown('---')
            st.title("➖ Remove Field")
            st.write("Remove an existing health data field.")
            name = st.selectbox("Name", list(st.session_state["health_data"].keys()))

            if st.button("Remove"):
                del st.session_state["health_data"][name]
                st.rerun()

        except Exception as e:
            st.error(f"Error processing health data: {str(e)}")

    chat_placeholder = st.empty()
    with chat_placeholder.container():
        messages = st.session_state.get('messages', [])
        for i, msg in enumerate(messages[1:]):
            if i % 2 == 0:
                content = msg.content.split("\nHere are my vitals:")[0]
                message(content, is_user=True, key=str(i) + '_user')
            else:
                message(msg.content, is_user=False, key=str(i) + '_ai')

        # clear chat history
        if st.button("Clear Chat"):
            st.session_state.messages = [
                SystemMessage(content="You are the best doctor; you can answer any question asked to you. You also like to give advice and motivate people to live a healthy life and you also have a good sense of humor. you also like to use emojis in your conversation."),
            ]
            st.rerun()

def main():
    init()
    ai_chat()

if __name__ == "__main__":
    main()
