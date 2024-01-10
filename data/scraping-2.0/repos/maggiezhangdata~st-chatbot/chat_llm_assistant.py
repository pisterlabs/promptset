from openai import OpenAI
import streamlit as st

st.title("A ChatBot That Really Listens to You")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
assistant_id = st.secrets["assistant_id"]

# Initialize or retrieve the thread_id in the session state
if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

if "show_thread_id" not in st.session_state:
    st.session_state.show_thread_id = False

# Layout for Copy Thread ID button and Disclaimer
col1, col2 = st.columns([2, 8])
with col1:
    if st.button("Copy Thread ID"):
        st.session_state.show_thread_id = True
with col2:
    with st.expander("ℹ️ Disclaimer"):
        st.caption(
            "We appreciate your engagement! Please note, this demo is designed to process a maximum of 10 interactions. Thank you for your understanding."
        )

# Display the thread ID
if st.session_state.show_thread_id:
    st.markdown("#### Thread ID")
    st.info(st.session_state.thread_id)
    st.caption("Please copy the above Thread ID")

# Initialize or retrieve the message history in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Displaying Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handling message input and response
max_messages = 20  # 10 iterations of conversation (user + assistant)
if len(st.session_state.messages) < max_messages:
    if user_input := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # Create a message in the thread
            message = client.beta.threads.messages.create(
                        thread_id=st.session_state.thread_id,
                        role="user",
                        content=user_input
                    )

            # Create and check run status
            run = client.beta.threads.runs.create(
                  thread_id=st.session_state.thread_id,
                  assistant_id=assistant_id,
                  instructions="Please address the user as test user. Your response should be exactly 10 words long."
                )

            # Wait until run is complete
            while True:
                run_status = client.beta.threads.runs.retrieve(
                          thread_id=st.session_state.thread_id,
                          run_id=run.id
                        )
                if run_status.status == "completed":
                    break

            # Retrieve and display messages
            messages = client.beta.threads.messages.list(
                    thread_id=st.session_state.thread_id
                    )

            full_response = messages.data[0].content[0].text.value
            message_placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
else:
    st.info(
        "Notice: The maximum message limit for this demo version has been reached..."
    )
