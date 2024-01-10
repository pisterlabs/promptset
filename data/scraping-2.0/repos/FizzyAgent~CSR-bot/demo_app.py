import streamlit as st
from iso3166 import countries

from api.driver.logic_service import run
from api.models.messages import Role, Message
from app.rendering import render_left_message, render_style, render_right_message
from app.states import (
    init_states,
    get_chat_messages,
    get_interface_messages,
    save_customer_message,
    get_all_companies,
    set_chat_settings,
    get_resource_loader,
    get_program_loader,
)
from api.models.settings import ChatSettings
from keys import OPENAI_API_KEY

MAX_MESSAGES = 50

st.set_page_config(page_title="CSR Bot Demo", layout="wide")

with st.sidebar:
    left, _, mid = st.columns((2, 0.1, 3))
right = st.container()

render_style()
init_states()

with left:
    st.markdown("**Settings**")
    company = st.selectbox(label="Select a Company", options=get_all_companies())
    countries = [c.name for c in countries]
    location = st.selectbox(
        label="Your Location",
        options=countries,
        index=countries.index("Singapore"),
    )
    set_chat_settings(company_name=company, location=location)
    resource_loader = get_resource_loader()
    all_resources = resource_loader.get_all_resources()
    resource_display = [
        "- " + x.replace(".txt", "").replace("_", " ").capitalize()
        for x in all_resources
    ]
    resource_display.sort()
    st.markdown(
        f"Here are the enquiries that CSR Bot can help for {company}:  \n"
        + "\n".join(resource_display)
        + "\n\nMore workflows and companies will be added soon!\n\n"
        "Disclaimer:  \n"
        "This app is for demonstration purposes. "
        "Do not enter your actual personal information."
    )
    settings = ChatSettings(
        company=company,
        location=location,
        resource_loader=resource_loader,
        program_loader=get_program_loader(),
    )
    st.markdown("-----")
    openai_key = st.text_input(
        label="OpenAI Key",
        value=OPENAI_API_KEY,
        help="Get a free key from https://platform.openai.com",
    )

with mid:
    st.markdown("### Behind the scenes of CSR Bot")
    interface_messages = get_interface_messages()
    for message in interface_messages:
        if message.role == Role.bot:
            render_right_message(delta=mid, message=message)
        elif message.role == Role.app:
            render_left_message(delta=mid, message=message)

with right:
    st.markdown("### Chat with CSR Bot")
    chat_messages = get_chat_messages()
    for message in chat_messages:
        if message.role == Role.customer:
            render_right_message(delta=st.container(), message=message)
        elif message.role == Role.bot:
            render_left_message(delta=st.container(), message=message)

    has_openai_key = len(openai_key) > 0
    user_input = st.chat_input(
        placeholder="Type your message here", max_chars=200, disabled=not has_openai_key
    )
    if not has_openai_key:
        st.markdown(
            "Please enter your OpenAI key under Settings to start chatting with CSR Bot!"
        )

    if user_input is not None and len((user_input := user_input.strip())) > 0:
        new_message = Message(role=Role.customer, text=user_input)
        save_customer_message(message=new_message)

terminated = False
messages = get_chat_messages()
interface_messages = get_interface_messages()
if (
    len(interface_messages) < MAX_MESSAGES
    and len(messages) > 0
    and messages[-1].role != Role.bot
):
    while (
        len(interface_messages) < MAX_MESSAGES
        and not terminated
        and messages[-1].role != Role.bot
    ):
        interface_messages = get_interface_messages()
        terminated = run(
            messages=interface_messages, settings=settings, openai_key=openai_key
        )
        messages = get_chat_messages()
    st.experimental_rerun()
