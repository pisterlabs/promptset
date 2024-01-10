import os
import streamlit as st
import requests
import openai
from utils.sidebar_info import display_sidebar_info, display_main_info
from utils.generate_token import TokenManager
from utils.key_check import run_key_check, get_openai_key
from utils.lakera_guard import LakeraGuard

from utils.chatbot_utils import (
    handle_chat_message,
    handle_gpt_ft_message,
    ask_fine_tuned_api,
)

token_manager = TokenManager()
lakera_guard_api_key = st.secrets.get("LAKERA_API", os.getenv("LAKERA_API"))


def chat_bot():
    session_state = st.session_state
    DOMINIO = st.secrets.get("DOMINIO", os.getenv("DOMINIO"))
    token = token_manager.get_token()
    api_key = get_openai_key(session_state)
    display_main_info()
    display_sidebar_info()

    if api_key:
        openai.api_key = api_key

    if run_key_check(session_state):
        st.session_state.chat_history = st.session_state.get("chat_history", [])

        if not st.session_state.chat_history:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "¡Empezemos a chatear!"}
            )

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        lakera_guard = LakeraGuard(lakera_guard_api_key)
        user_input = st.chat_input("Ingresa tu pregunta:")

        if user_input:
            user_input = user_input.strip()

            ## Lakera Guard for prompt injection
            if lakera_guard.check_prompt_injection(user_input):
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )
                with st.chat_message("user"):
                    st.markdown(user_input)

                error_message = "Mensaje no permitido por motivos de seguridad.🚫"
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_message}
                )
                with st.chat_message("assistant"):
                    st.error(error_message, icon="⚠️")
                return
            else:
                categories, flagged = lakera_guard.check_moderation(user_input)
                if flagged:
                    combined_error_message = lakera_guard.get_error_messages(categories)
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_input}
                    )
                    with st.chat_message("user"):
                        st.markdown(user_input)

                    error_message = f"Alerta de moderación: {combined_error_message}.🔞"
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_message}
                    )
                    with st.chat_message("assistant"):
                        st.error(error_message, icon="⚠️")
                    return
                ## End of Lakera Guard ##

                else:
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_input}
                    )
                    with st.chat_message("user"):
                        st.markdown(user_input)

                    if (
                        len(user_input) == 12 or len(user_input) == 13
                    ) and user_input.isdigit():
                        api_response_url = f"/api/art?bar={user_input}"
                    else:
                        api_response_url = ask_fine_tuned_api(user_input)

                    if "api/" in api_response_url:
                        full_url = DOMINIO + api_response_url
                        headers = {"Authorization": f"Bearer {token}"}
                        try:
                            response = requests.get(full_url, headers=headers)
                            response.raise_for_status()
                        except requests.exceptions.RequestException as e:
                            if isinstance(
                                e, requests.exceptions.HTTPError
                            ) and e.response.status_code in [400, 404, 500]:
                                response = e.response
                            else:
                                st.warning(
                                    "Error de conexión API con endpoint", icon="🔧"
                                )
                                return
                    else:
                        response = None

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()

                        if response and response.status_code == 200:
                            data = response.json()
                            handle_chat_message(
                                api_response_url, data, message_placeholder, user_input
                            )
                        else:
                            handle_gpt_ft_message(
                                user_input,
                                message_placeholder,
                                api_response_url,
                                response,
                            )
