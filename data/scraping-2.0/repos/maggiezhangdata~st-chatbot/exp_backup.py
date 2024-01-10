from openai import OpenAI
import streamlit as st
import time
import re  # Import regular expressions

st.title("聊天机器人")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
assistant_id = st.secrets["asst_oD8fP9bP1DiBEMlNM0ss7pDE"]



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
    with st.expander("ℹ️ 声明"):
        st.caption(
            "我们感谢您的参与！ 请注意，此机器人最多可处理 10 轮对话。 感谢您的理解。"
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
max_messages = 200  # 10 iterations of conversation (user + assistant)
if len(st.session_state.messages) < max_messages:
    if user_input := st.chat_input("最近还好吗？"):
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
                  # instructions="Forget all your previous instructions, and follow strictly the following 3 rules: 1. when given the same input, always output the same response. set your temperature parameter in your chat completion function to be 0.1. 2. When engaging in a conversation, your primary goal is to foster elaboration by posing a question 3. When presenting solutions or suggestions, offer three succinct bullet points, with a total word count of fewer than 180 Chinese characters."
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





            def format_response(response):
                """
                Formats the response to handle bullet points and new lines.
                Targets both ordered (e.g., 1., 2.) and unordered (e.g., -, *, •) bullet points.
                """
                # Split the response into lines
                lines = response.split('\n')
                
                formatted_lines = []
                for line in lines:
                    # Check if the line starts with a bullet point (ordered or unordered)
                    if re.match(r'^(\d+\.\s+|[-*•]\s+)', line):
                        formatted_lines.append('\n' + line)
                    else:
                        formatted_lines.append(line)

                # Join the lines back into a single string
                formatted_response = '\n'.join(formatted_lines)

                return formatted_response.strip()



            # #------ adding speed variation for english --------
            # words = full_response.split()
            # speed = 2
            # delay_per_word = 1.0 / speed
            # displayed_message = ""
            # for word in words:
            #     displayed_message += word + " "
            #     formatted_message = format_response(displayed_message) # Format for bullet points
            #     message_placeholder.markdown(formatted_message)
            #     time.sleep(delay_per_word)  # Wait for calculated delay time

            # #------ end speed variation for english --------

            #------ adding speed variation for Chinese --------
            full_response = format_response(full_response)  # Format for bullet points
            chars = list(full_response)
            speed = 20  # Display 5 Chinese characters per second
            delay_per_char = 1.0 / speed
            displayed_message = ""
            for char in chars:
                displayed_message += char
                message_placeholder.markdown(displayed_message)
                time.sleep(delay_per_char)  # Wait for calculated delay time

            #------ end speed variation for Chinese --------



            # message_placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
else:
    st.info(
        "注意：已达到此演示版本的最大消息限制..."
    )
