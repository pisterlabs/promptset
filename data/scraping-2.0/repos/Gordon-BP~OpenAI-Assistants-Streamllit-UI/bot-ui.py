import streamlit as st
from nanoid import generate
from typing import Dict
from util.pydantic_classes import (
    Event,
    BotMessageTypes,
    BotMessage,
    BotButtonMessage,
    BotImageMessage,
    BotTextMessage,
    Choice,
)
from util.make_elements import (
    makeButtons,
    makeImage,
    makeMarkdown,
    makeText,
)
from util.generate_image import generateImage
import json
import time
import openai
import os
from dotenv import load_dotenv
from util.logger import logger

load_dotenv()

if os.environ.get("OPENAI_ASSISTANT_NAME"):
    st.title(os.environ.get("OPENAI_ASSISTANT_NAME"))
if os.environ.get("BOT_DESCRIPTION"):
    st.markdown(os.environ.get("BOT_DESCRIPTION"))
client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

def getBotResponse(userEvent: Event) -> Event:
    """
    Retrieves the bot response for a given user event.

    This function uses the new assistants endpoints to get bot response.

    Args:
    - userEvent: An Event object that contains the user's input.

    Returns:
    - Event: An Event object containing the bot's response.
    """
    # First we need to ensure the run state is ready to receive a new event
    run = client.beta.threads.runs.retrieve(
        run_id=st.session_state.runId,
        thread_id=st.session_state.threadId,
    )
    while run.status == "in_progress":
        logger.debug("Run in progress, waiting....")
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            run_id=st.session_state.runId,
            thread_id=st.session_state.threadId,
        )
        logger.debug(f"Current run status is {run.status}")
    event_dict = userEvent.model_dump()
    # Then, we actually need to send the user reply to the model
    if run.status == "failed":
        # Early termination because run failure, usually because of rate limiting
        logger.error(f"Run failed: {run.last_error.code}. {run.last_error.message}")
        event_dict["botReply"] = [
            BotMessage(
                type="text",
                payload=BotTextMessage(
                    text=f"Sorry, there was an error in the chat: {run.last_error.code}. {run.last_error.message}",
                    useMarkdown=True,
                ),
            ),
        ]
        event_dict["direction"] = "outgoing"
        botEvent = Event(**event_dict)
        return botEvent
    
    if run.status == "requires_action": 
        # The last message was a tool use, so we have to submit user response as a tool call output
        logger.debug("Submitting user input as tool output")
        event_dict["userInput"] = event_dict["payload"]["text"]
        client.beta.threads.runs.submit_tool_outputs(
            run_id=st.session_state.runId,
            thread_id=st.session_state.threadId,
            tool_outputs=[
                {
                    "tool_call_id": run.required_action.submit_tool_outputs.tool_calls[
                        0
                    ].id,
                    "output": str(event_dict["payload"]),
                }
            ],
        )
        # After submitting the userResponse, wait for processing to complete
        run = client.beta.threads.runs.retrieve(
            run_id=st.session_state.runId, thread_id=st.session_state.threadId
        )
    if run.status == "completed":
        # The last message was normal text, so we need to add a new text message
        logger.debug("Adding text message to thread")
        client.beta.threads.messages.create(
            thread_id=st.session_state.threadId,
            content=str(event_dict['payload']),
            role="user",
        )
        # This is where we start the next run?
        logger.debug(f"Starting new run...")
        run = client.beta.threads.runs.create(
            #run_id=st.session_state.runId,
            assistant_id=st.session_state.assistantId,
            thread_id=st.session_state.threadId
            )
        st.session_state.runId = run.id
        run = client.beta.threads.runs.retrieve(
            run_id=st.session_state.runId,
            thread_id=st.session_state.threadId,
        )
    logger.debug(f"Run is now: {run}")
    # Now that we have sent things to the API, we wait for the response
    while run.status == "in_progress":
        logger.debug("Submitted new user input, waiting....")
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            run_id=st.session_state.runId,
            thread_id=st.session_state.threadId,
        )
        logger.debug(f"Current run status is {run.status}")

    logger.info("Received payload back from the assistant!")
    if run.status == "requires_action":
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            logger.debug(f"Processing tool call {tool_call.function.name}")
            if tool_call.function.name == "show_buttons":
                logger.debug("Showing buttons...")
                args = json.loads(tool_call.function.arguments)
                text, choices = args["text"], args["choices"]
                event_dict["botReply"].append(
                    BotMessage(
                        type="button",
                        payload=BotButtonMessage(
                            text=text,
                            choices=[
                                Choice(label=n["label"], value=n["value"])
                                for n in choices
                            ],
                            active=True,
                        ),
                    ),
                )
            if tool_call.function.name == "generate_image":
                logger.debug("Generating image...")
                args = json.loads(tool_call.function.arguments)
                prompt = args['prompt']
                event_dict['botReply'].append(
                    BotMessage(
                        type="image",
                        payload=BotImageMessage(
                            url=generateImage(prompt)
                        )
                    )
                )
                # Tell the API that the image was successfully generated
                client.beta.threads.runs.submit_tool_outputs(
                    run_id=st.session_state.runId,
                    thread_id=st.session_state.threadId,
                    tool_outputs=[
                        {
                            "tool_call_id": run.required_action.submit_tool_outputs.tool_calls[0].id,
                            "output": "{\"status\":200}",
                        }
                    ],
                )
        
    if run.status == "completed":
        logger.debug("No required actions, sending messages...")
        messages = client.beta.threads.messages.list(
            thread_id=st.session_state.threadId
        )
        event_dict["botReply"] = [
            BotMessage(
                type="text",
                payload=BotTextMessage(
                    text=messages.data[0].content[0].text.value,
                    useMarkdown=True,
                ),
            ),
        ]
    if run.status == "failed":
        # Early termination because run failure, usually because of rate limiting
        logger.error(f"Run failed: {run.last_error.code}. {run.last_error.message}")
        event_dict["botReply"] = [
            BotMessage(
                type="text",
                payload=BotTextMessage(
                    text=f"Sorry, there was an error in the chat: {run.last_error.code}. {run.last_error.message}",
                    useMarkdown=True,
                ),
            ),
        ]

    event_dict["direction"] = "outgoing"
    botEvent = Event(**event_dict)
    return botEvent


def deactivateButtons() -> None:
    """
    Deactivates buttons in the most recent message in the session state.

    This function iterates over the last event's bot replies and deactivates any buttons found.
    """
    prevEvent = st.session_state.messages[-1]["content"]
    for reply in prevEvent.botReply:
        if reply.type == BotMessageTypes.button:
            reply.payload.active = False
    st.session_state.messages[-1]["content"] = prevEvent


def makeUserMessage(userInput: Dict[str, str] = {}) -> Event:
    """
    Creates a user message event from the given input and updates the session state.

    Args:
    - userInput: A dictionary containing the user's input text, if available.

    Returns:
    - Event: An Event object representing the user's message.
    """
    if not userInput and ("userInput" in st.session_state):
        userInput = {"type": "text", "text": st.session_state['userInput']}
        logger.info(f"User pressed button: {st.session_state['userInput']}")
    else:
        logger.info(f"User said: {userInput}")
    deactivateButtons()
    userEvent = Event(
        userId=st.session_state.userId,
        conversationId=st.session_state.conversationId,
        direction="incoming",
        payload=userInput,
    )
    st.session_state.messages.append({"role": "user", "content": userEvent})
    return userEvent

def init_session_state():
    """
    Initializes the Streamlit session state with necessary values.

    This includes creating a new thread and run for the OpenAI assistant,
    generating a new user ID and conversation ID, and preparing the first message.
    """
    logger.debug("Initializing streamlit session...")
    thread = client.beta.threads.create(messages=[{"role": "user", "content": "Hello"}])
    assistant = client.beta.assistants.retrieve(assistant_id=os.environ.get("OPENAI_ASSISTANT_ID"))
    if "threadId" not in st.session_state:
        st.session_state.threadId = thread.id
    if "assistantId" not in st.session_state:
        st.session_state.assistantId = assistant.id
    if "runId" not in st.session_state:
        run = client.beta.threads.runs.create(
            thread_id=thread.id, assistant_id=assistant.id
        )
        st.session_state.runId = run.id
    if "userId" not in st.session_state:
        st.session_state.userId = generate(size=12)
    if "conversationId" not in st.session_state:
        st.session_state.conversationId = generate(size=14)
    # init the OpenAI agent run
    run = client.beta.threads.runs.retrieve(
        run_id=st.session_state.runId, thread_id=st.session_state.threadId
    )
    # Wait for run to finish
    while run.status == "in_progress":
        logger.debug("Run initializing, waiting...")
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            run_id=st.session_state.runId, thread_id=st.session_state.threadId
        )
        logger.debug(f"Run is {run.status}")
    # Add the first bot message
    if "messages" not in st.session_state:
        if run.status == "requires_action":
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                logger.debug(f"Processing tool call {tool_call.function.name}")
                if tool_call.function.name == "show_buttons":
                    logger.debug("Showing buttons...")
                    args = json.loads(tool_call.function.arguments)
                    text, choices = args["text"], args["choices"]
                    st.session_state.messages = [{
                        "role":"assistant",
                        "content":Event(
                            userId=st.session_state.userId,
                            conversationId=st.session_state.conversationId,
                            direction="outgoing",
                            botReply=[BotMessage(
                            type="button",
                            payload=BotButtonMessage(
                                text=text,
                                choices=[
                                    Choice(label=n["label"], value=n["value"])
                                    for n in choices
                                ],
                                active=True,
                            ),
                        )],
                    )}]
                if tool_call.function.name == "generate_image":
                    logger.debug("Generating image...")
                    args = json.loads(tool_call.function.arguments)
                    prompt = args['prompt']
                    st.session_state.messages = [
                        {
                            "role":"assistant",
                            "content":Event(
                                userId=st.session_state.userId,
                                conversationId=st.session_state.conversationId,
                                direction="outgoing",
                                botReply=[BotMessage(
                            type="image",
                            payload=BotImageMessage(
                                url=generateImage(prompt)
                            )
                        )]
                    )}]
                # Tell the API that the image was successfully generated
                client.beta.threads.runs.submit_tool_outputs(
                    run_id=st.session_state.runId,
                    thread_id=st.session_state.threadId,
                    tool_outputs=[
                        {
                            "tool_call_id": run.required_action.submit_tool_outputs.tool_calls[0].id,
                            "output": "{\"status\":200}",
                        }
                    ],
                )
        if run.status == "failed":
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": Event(
                        userId=st.session_state.userId,
                        conversationId=st.session_state.conversationId,
                        direction="outgoing",
                        botReply=[
                            BotMessage(
                                type="text",
                                payload=BotTextMessage(
                                    text=f"There was an error starting the chat: {run.last_error.code}. {run.last_error.message}",
                                    useMarkdown=True,
                                ),
                            )
                        ],
                    ),
                }
            ]
        if run.status == "completed":
            logger.debug("Message with no buttons")
            messages = client.beta.threads.messages.list(
                thread_id=st.session_state.threadId
                )
            st.session_state.messages = [
                {
                "role":"assistant",
                "content": Event(
                    userId=st.session_state.userId,
                    conversationId=st.session_state.conversationId,
                    direction="outgoing",
                    botReply=[
                        BotMessage(
                            type="text",
                            payload=BotTextMessage(
                                text=messages.data[0].content[0].text.value,
                                useMarkdown=True,
                    ),
                )]
                )
            }]


# Initialize messages with welcome message
if __name__ == "__main__":
    if "messages" not in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Loading quiz..."):
                init_session_state()
                st.rerun()
    # Write messages to app
    for message in st.session_state.messages:
        if message["role"] != "user":
            with st.chat_message("assistant"):
                for reply in message["content"].botReply:
                    if reply.type == BotMessageTypes.button:
                        logger.debug("Writing bot button message to chat...")
                        makeButtons(reply.payload, makeUserMessage)
                    if reply.type == BotMessageTypes.text:
                        if reply.payload.useMarkdown:
                            logger.debug("Writing bot markdown message to chat...")
                            makeMarkdown(reply.payload)
                        else:
                            logger.debug("Writing bot text message to chat...")
                            makeText(reply.payload)
        else:
            with st.chat_message("user"):
                logger.debug("Writing user message to chat...")
                st.markdown(message["content"].payload["text"])
    if st.session_state.messages[-1]["role"] != "assistant":
        logger.debug("Processing user input...")
        with st.chat_message("assistant") as msg:
            with st.spinner("Thinking..."):
                botEvent = getBotResponse(st.session_state.messages[-1]['content'])
                st.session_state.messages.append(
                    {"role": "assistant", "content": botEvent}
                )
                for reply in botEvent.botReply:
                    if reply.type == BotMessageTypes.button:
                        logger.debug("Writing bot button message to chat...")
                        makeButtons(reply.payload, makeUserMessage)
                    if reply.type == BotMessageTypes.text:
                        if reply.payload.useMarkdown:
                            logger.debug("Writing bot markdown message to chat...")
                            makeMarkdown(reply.payload)
                        else:
                            logger.debug("Writing bot text message to chat...")
                            makeText(reply.payload)
    logger.debug("Waiting for user input...")
    prompt = st.chat_input(
        "Type your response here", key="userInput", on_submit=makeUserMessage,
    )
