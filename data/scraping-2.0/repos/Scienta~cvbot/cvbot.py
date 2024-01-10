import datetime
import os
import sys
import time

import openai.types.beta
import scienta.cvpartner_api
import streamlit as st
import streamlit.logger
from openai import OpenAI, OpenAIError
from openai.types.beta.threads import required_action_function_tool_call, Run, ThreadMessage, \
    run_submit_tool_outputs_params, MessageContentText
from streamlit.delta_generator import DeltaGenerator

import scienta.cv_db
import scienta.openai_functions as fs

example_prompts = [
    # "List all people in Scienta. Format it as a Markdown output and sort the list of names.",
    "Lage en sortert liste over alle ansatte i Scienta.",
    # "How many people do you have in your database?",
    "Hvor mange personer har du i databasen din?",
    # "Tell me about Trygve.",
    "Lag en kort oppsummering om Trygve",
]


@st.cache_data()
def get_assistant() -> openai.types.beta.Assistant:
    assistant_id = os.getenv("ASSISTANT_ID")
    if assistant_id is None:
        st.error("Missing required environment variable: ASSISTANT_ID")
        st.stop()

    logger.info("Loading assistant")
    assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
    logger.info(f"Assistant: {assistant.model_dump_json(indent=2)}")
    return assistant


def get_thread() -> openai.types.beta.Thread:
    thread = create_thread()

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = thread.id
        append_message("assistant", f"Created thread *{thread.id}*")

    return thread


@st.cache_data()
def create_thread() -> openai.types.beta.Thread:
    logger.info("Creating thread")
    thread = client.beta.threads.create()
    logger.info(f"Created thread: {thread.id}")
    show_json(thread)
    return thread


@st.cache_resource
def get_cvpartner_api():
    cvpartner_api_key = os.getenv("CVPARTNER_API_KEY")
    if cvpartner_api_key is None:
        st.error("CVPARTNER_API_KEY not set")
        st.stop()

    api_client = scienta.cvpartner_api.ApiClient()
    api_client.set_default_header("Authorization", f"Bearer {cvpartner_api_key}")
    api_client.configuration.server_variables["subdomain"] = "scienta"
    return scienta.cvpartner_api.DefaultApi(api_client=api_client)


def show_json(obj):
    s = obj.model_dump_json(indent=2)
    logger.debug(s)


def run_tool_call(function: required_action_function_tool_call.Function):
    m = f"Calling function {function.name} with arguments: {function.arguments}"
    logger.info(m)
    append_message("ai", m)

    s = fs.run_tool_call(function.name, function.arguments)

    logger.info(f"Response: {s}")

    return s


def process_message(s: str):
    run, message = run_message(s)

    while True:
        if run.required_action is None:
            break

        logger.info("Got a required action")
        show_json(run.required_action)

        tcs = run.required_action.submit_tool_outputs.tool_calls
        responses = []
        for tc in tcs:
            s = run_tool_call(tc.function)
            responses.append(run_submit_tool_outputs_params.ToolOutput(
                tool_call_id=tc.id,
                output=s,
            ))

        logger.info("Submitting tool outputs")
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=get_thread().id,
            run_id=run.id,
            tool_outputs=responses,
        )
        run = wait_on_run(run)

    messages = client.beta.threads.messages.list(
        thread_id=get_thread().id, order="asc", after=message.id
    )
    logger.info(f"message page size {len(messages.data)}, json:")
    show_json(messages)

    for m in messages.data:
        for content in m.content:
            if isinstance(content, MessageContentText):
                txt: MessageContentText = content
                append_message("ai", txt.text.value)
            else:
                logger.info(f"Unsupported content type: {content.type}")


def wait_on_run(run):
    while run.status == "queued" or run.status == "in_progress":
        logger.info(f"Waiting for run: {run.status}")

        run = client.beta.threads.runs.retrieve(
            thread_id=get_thread().id,
            run_id=run.id,
        )
        time.sleep(0.5)

    logger.info(f"Run is ready: {run.status}")

    return run


def run_message(q) -> tuple[Run, ThreadMessage]:
    logger.info("Creating message")
    message = client.beta.threads.messages.create(
        thread_id=get_thread().id,
        role="user",
        content=q,
    )
    logger.info(f"Message id {message.id}")
    show_json(message)

    functions = [
        fs.get_resume_by_id,
        fs.get_people,
        fs.get_resume,
    ]

    logger.info("Creating run")
    run = client.beta.threads.runs.create(
        thread_id=get_thread().id,
        assistant_id=get_assistant().id,
        tools=[f.to_tool() for f in functions],
    )
    show_json(run)

    run = wait_on_run(run)
    show_json(run)

    return run, message


def append_message(role: str, content: str):
    st.session_state.messages.append(
        {"role": role, "content": content}
    )
    messages_container.chat_message(role).write(content)


try:
    client = OpenAI()
except OpenAIError:
    msg = "Could not configure OpenAI API. Are you missing the OPENAI_KEY environment variable?"
    print(msg, file=sys.stderr)
    st.error(msg)
    st.stop()

logger = streamlit.logger.get_logger(__name__)

logger.info("evaluating!")

scienta.openai_functions.client = scienta.cv_db.CvPartnerClient(get_cvpartner_api())
scienta.openai_functions.logger = streamlit.logger.get_logger("openai_functions")
scienta.cv_db.logger = streamlit.logger.get_logger("cv_db")

# We want this early, so it is loaded before anything is written to the user
get_assistant()

messages_container: DeltaGenerator = st.empty()


def run_app():
    def submit_message():
        s = st.session_state.input_text

        append_message("user", s)
        logger.info(f"Sending message: {s}")

        process_message(s)

    st.title("Welcome to Scienta's CvBot!")

    assistant = get_assistant()

    with st.expander(label="About the assistant"):
        st.markdown(f"* Id: {assistant.id}")
        st.markdown(f"* Model: {assistant.model}")
        st.markdown(f"* Name: {assistant.name}")
        instructions = "\n".join([f"> {l}" for l in assistant.instructions.splitlines()])
        st.markdown(f"Instructions:\n{instructions}")

    with st.expander(label="Example prompts", expanded=True):
        for p in example_prompts:
            st.write(f"* {p}")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    global messages_container
    messages_container = st.container()

    for message in st.session_state.messages:
        messages_container.chat_message(message["role"]).markdown(message["content"])

    if st.chat_input("How many people do you have in your database?",
                     key="input_text"):
        submit_message()


run_app()
