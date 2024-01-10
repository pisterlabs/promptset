import json
import time
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

#######################################
# PREREQUISITES
#######################################

st.set_page_config(
    page_title="Strum Buddy",
    page_icon="strum-buddy-logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

assistant_id = st.secrets["OPENAI_ASSISTANT_ID"]

assistant_state = "assistant"
thread_state = "thread"
conversation_state = "conversation"
last_openai_run_state = "last_openai_run"

user_msg_input_key = "input_user_msg"

#######################################
# SESSION STATE SETUP
#######################################

if (assistant_state not in st.session_state) or (thread_state not in st.session_state):
    st.session_state[assistant_state] = client.beta.assistants.retrieve(assistant_id)
    st.session_state[thread_state] = client.beta.threads.create()

if conversation_state not in st.session_state:
    st.session_state[conversation_state] = []

if last_openai_run_state not in st.session_state:
    st.session_state[last_openai_run_state] = None

#######################################
# HELPERS
#######################################


def get_assistant_id():
    return st.session_state[assistant_state].id


def get_thread_id():
    return st.session_state[thread_state].id


def get_run_id():
    return st.session_state[last_openai_run_state].id


def on_text_input(status_placeholder):
    """Callback method for any chat_input value change
    """
    
    if st.session_state[user_msg_input_key] == "":
        return

    client.beta.threads.messages.create(
        thread_id=get_thread_id(),
        role="user",
        content=st.session_state[user_msg_input_key],
    )
    st.session_state[last_openai_run_state] = client.beta.threads.runs.create(
        assistant_id=get_assistant_id(),
        thread_id=get_thread_id(),
    )

    completed = False

    # Polling
    with status_placeholder.status("OK! One minute while I assemble all the resources you'll need.") as status_container:
        #st.write(f"Launching run {get_run_id()}")
        st.write(f"Strum Buddy is Working...")
        while not completed:
            run = client.beta.threads.runs.retrieve(
                thread_id=get_thread_id(),
                run_id=get_run_id(),
            )

            if run.status == "requires_action":
                tools_output = []
                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    f = tool_call.function
                    print(f)
                    f_name = f.name
                    f_args = json.loads(f.arguments)

                    st.write(f"Launching function {f_name} with args {f_args}")
                    tool_result = tool_to_function[f_name](**f_args)
                    tools_output.append(
                        {
                            "tool_call_id": tool_call.id,
                            "output": tool_result,
                        }
                    )
                st.write(f"Will submit {tools_output}")
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=get_thread_id(),
                    run_id=get_run_id(),
                    tool_outputs=tools_output,
                )

            if run.status == "completed":
                #st.write(f"Completed run {get_run_id()}")
                status_container.update(label="Strum Buddy is Done!", state="complete")
                completed = True
                components.html(
    f"""
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
    """,
    height=0
)

            else:
                time.sleep(0.1)

    st.session_state[conversation_state] = [
        (m.role, m.content[0].text.value)
        for m in client.beta.threads.messages.list(get_thread_id()).data
    ]


def on_reset_thread():
    client.beta.threads.delete(get_thread_id())
    st.session_state[thread_state] = client.beta.threads.create()
    st.session_state[conversation_state] = []
    st.session_state[last_openai_run_state] = None
    

#######################################
# SIDEBAR
#######################################

with st.sidebar:
    st.header("Debug")
    st.write(st.session_state.to_dict())

    st.button("Reset Thread", on_click=on_reset_thread)


#######################################
# MAIN
#######################################

#CSS to hide developer options when deployed
#Comment out when developing locally to reveal debugging tools
st.markdown(
    """
    <style>
    header {display: none !important}
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    [data-testid="collapsedControl"] {
        display: none
    }
    .st-emotion-cache-z5fcl4 {padding-top: 0 !important}
    .viewerBadge_text__1JaDK {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



left_col, right_col = st.columns(2)

with left_col:

    with st.container():
            st.image("strum-buddy-logo.png", use_column_width="auto")


with right_col:
            
            st.subheader("Don't you want Strum Buddy to love?")
            st.write('Strum Buddy is an intelligent assistant who can help you locate all of the online resources needed to learn how to play new songs on guitar.')
            st.write('Simply provide a song title and artist and Strum Buddy will provide you with links to video tutorials and other helpful information from a variety of different websites.')
        
            for role, message in st.session_state[conversation_state]:
                with st.chat_message(role):
                    st.write(message)
            

status_placeholder = st.empty()
st.chat_input(
                placeholder="Provide a song name and artist",
                key=user_msg_input_key,
                on_submit=on_text_input,
                args=(status_placeholder,),
            )



