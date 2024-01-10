import openai
import streamlit as st
import os
from utils import (
    num_tokens_from_messages,
    num_tokens_from_output,
    make_converstaion_name_id,
)
import database
import time


def create_chat_interface():
    st.title("ChatGPT 🤖")
    st.text(f"model: {st.session_state.model}")

    openai.api_key = os.environ["OPENAI_API_KEY"]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # load previous conversation from db
    con_id = get_cid_from_url()
    if con_id is not None:
        conversation = database.get_conversation(con_id)

        if conversation is None:
            st.toast("Conversation was not found in history.", icon="🚨")
            time.sleep(1)
            st.toast("Opening a new chat.", icon="🚨")
            time.sleep(1)
            st.session_state.messages == []
            st.experimental_set_query_params()

        else:
            st.session_state.messages = conversation.messages
            st.session_state.tokens_used = conversation.total_tokens
            st.session_state.model = conversation.model

            con = {}
            con["conversation_id"] = conversation.id
            con["conversation_name"] = conversation.conversation_name

            st.session_state.conversation = con

    if st.session_state.messages == []:
        # initilize conversation id and name, will be produced later
        # when there is first prompt available
        st.session_state.conversation = None

        # initilizing tokens
        tokens_used = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # save tokens_used in session
        st.session_state.tokens_used = tokens_used

    for message in st.session_state.messages:
        # write previous messages in the chat from session store
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # make conversation id and name based on first prompt and create db entry
        if st.session_state.conversation is None:
            conversation = make_converstaion_name_id(prompt)
            st.session_state.conversation = conversation
            database.create_new_conversation(
                conversation_id=conversation["conversation_id"],
                conversation_name=conversation["conversation_name"],
                model_name=st.session_state["model"],
                system_msg=None,
                messages=[],
                total_tokens=tokens_used,
                total_price=0,
                user_name=st.session_state["username"],
            )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            all_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            input_tokens = num_tokens_from_messages(all_messages)

            responses = openai.ChatCompletion.create(
                model=st.session_state["model"],
                messages=all_messages,
                stream=True,
                # temperature=0,
            )

            for response in responses:
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        response_message = {"role": "assistant", "content": full_response}

        st.session_state.messages.append(response_message)

        output_tokens = num_tokens_from_output(response_message)

        # Estimate cost

        tokens = input_tokens + output_tokens

        # sum total tokens per chat, note that every
        # query adds tokens per chat
        tokens_used = st.session_state.tokens_used
        tokens_used["prompt_tokens"] += input_tokens
        tokens_used["completion_tokens"] += output_tokens
        tokens_used["total_tokens"] += tokens

        # write tokens back into session
        st.session_state.tokens_used = tokens_used

        prices = st.session_state.all_models[st.session_state.model]["price"]

        cost = (
            tokens_used["prompt_tokens"] / 1000.0 * prices["input"]
            + tokens_used["completion_tokens"] / 1000.0 * prices["output"]
        )
        st.text(f"Estimated cost: ${cost:.3f} ({tokens_used['total_tokens']} tokens)")

        database.update_conversation(
            conversation_id=st.session_state["conversation"]["conversation_id"],
            messages=st.session_state["messages"],
            total_tokens=tokens_used,
            total_price=cost,
        )

        database.update_credit_used(st.session_state["username"])


def create_sidebar(authenticator):
    models_list = list(st.session_state["all_models"].keys())

    with st.sidebar:
        # create login-logut
        user = st.session_state["name"]
        user_name = st.session_state["username"]
        st.write(f"Welcome *{user}*")

        if authenticator:
            authenticator.logout("Logout", "main", key="unique_key")

        credit_used = database.get_user_credit_used(user_name)
        st.markdown(f"Credit Used: `${credit_used:.3f}`")

        # create new chat button
        st.divider()
        st.markdown("### Creat a New Chat")

        # create a model choice radio
        # check if conversation from history is loaded
        cid = get_cid_from_url()

        # if convestation loded model cannot be changed
        if cid:
            st.session_state.radio_disabled = True
        else:
            st.session_state.radio_disabled = False

        model = st.radio(
            "Choose a model",
            models_list,
            disabled=st.session_state.radio_disabled,
        )

        st.session_state["model"] = model

        new_chat = st.button("New Chat")
        if new_chat:
            # reset the chat messages
            st.session_state.messages = []
            # strip query params
            st.experimental_set_query_params()
            time.sleep(0.1)
            st.experimental_rerun()


def create_history():
    user_name = st.session_state["username"]
    with st.sidebar:
        st.divider()
        st.markdown("### History")
        st.text("Only last 10 entries")

        con_id = get_cid_from_url()

        for conversation in database.get_user_conversations(user_name):
            conversation_name = conversation.conversation_name
            link = f"<a href='/?cid={conversation.id}' target='_self'>`{conversation_name}`</a>"
            costs = f"`${conversation.total_price:.3f}`"
            markdown_string = link + costs

            # bold the chosen conversation
            if conversation.id == con_id:
                markdown_string = f"**{markdown_string}**"

            st.markdown(
                markdown_string,
                unsafe_allow_html=True,
            )


def get_cid_from_url():
    query_params = st.experimental_get_query_params()
    con_id = query_params.get("cid", [None])[0]

    return con_id


def create_users_from_auth_config(auth_config):
    users = auth_config["credentials"]["usernames"]

    for user in users:
        database.create_new_user(user, users[user]["email"])


def create_user(user_name, email):
    database.create_new_user(user_name, email)
