from openai import OpenAI
import streamlit as st
import time
import re  # Import regular expressions

st.title("聊天机器人")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
assistant_id = st.secrets["assistant_id_a2_20"]
speed = 100




if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

if "show_thread_id" not in st.session_state:
    st.session_state.show_thread_id = False

if "first_message_sent" not in st.session_state:
    st.session_state.first_message_sent = False

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
st.sidebar.markdown("#### 完成对话后，复制对话编号")
st.sidebar.info(st.session_state.thread_id)
st.sidebar.caption("请复制上述对话编号。")
    
# Handling message input and response
max_messages = 15  # 10 iterations of conversation (user + assistant)

if len(st.session_state.messages) < max_messages:
    
    user_input = st.chat_input("")
    if not st.session_state.first_message_sent:
        st.markdown(
            "您可以通过复制粘贴<br>"
            "<span style='color: #8B0000;'>我最近很心烦，请告诉我该怎么办？</span><br>"
            "到下面👇🏻的对话框，开启和聊天机器人的对话，寻求建议和帮助。", unsafe_allow_html=True
        )
    if user_input:
        st.session_state.first_message_sent = True
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
            # speed = 20  # Display 5 Chinese characters per second
            delay_per_char = 1.0 / speed
            displayed_message = ""
            for char in chars:
                displayed_message += char
                message_placeholder.markdown(displayed_message)
                time.sleep(delay_per_char)  # Wait for calculated delay time

            #------ end speed variation for Chinese --------


            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

else:

    if user_input:= st.chat_input(""):
        with st.chat_message("user"):
            st.markdown(user_input)
        

    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.info(
                "已达到此聊天机器人的最大对话限制，请复制侧边栏对话编号。将该对话编号粘贴在下面的文本框中。"
            )
    st.chat_input(disabled=True)

    # # Button to copy thread ID
    # if st.button("复制thread_id"):
    #     st.session_state.show_thread_id = True

    # # When thread ID is shown, update the flag to hide the input box
    # if st.session_state.get('show_thread_id', False):
    #     st.session_state['thread_id_shown'] = True  # Set the flag to hide the input box
    #     st.markdown("#### Thread ID")
    #     st.info(st.session_state.thread_id)
    #     st.caption("请复制以上文本框中的thread_id。")



#----------------------------------------------
# else:
#     user_input = st.chat_input("最近还好吗？")
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     # with st.chat_message("user"):
#     #     st.markdown(user_input)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         message_placeholder.info(
#             "注意：已达到此聊天机器人的最大消息限制，请点击复制thread_id按钮，复制thread_id。将该thread_id粘贴在下一页的回答中。"
#         )
    

#     if st.button("复制thread_id"):
#         st.session_state.show_thread_id = True

#     if st.session_state.show_thread_id:
#         st.markdown("#### Thread ID")
#         st.info(st.session_state.thread_id)
#         st.caption("请复制以上文本框中的thread_id。")




