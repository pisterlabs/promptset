import openai
import weaviate
import streamlit as st


def connect_to_openai(openai_api_key: str) -> None:
    """Try to connect to OpenAI using the API key.
    Set the state variable OPENAI_STATUS based on the outcome
    """
    try:
        openai.api_key = openai_api_key
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[dict(role="user", content="hello")],
        )
    except Exception as e:
        st.session_state["OPENAI_STATUS"] = "error", e
        return
    
    st.session_state["OPENAI_STATUS"] = "success", None


def user_auth_openai():
    """Container to enter OpenAI API key"""
    form = st.form("openai_auth_form")

    openai_api_key = form.text_input(
        "OpenAI API key", 
        type="password",
        help="[docs](https://platform.openai.com/docs/api-reference/introduction)",
        key="openai_api_key_main",
    )

    if form.form_submit_button("Authenticate"):
        connect_to_openai(openai_api_key)


def openai_connection_status():
    """Message about the success/failure of the OpenAI connection"""
    openai_status, openai_message = st.session_state.get("OPENAI_STATUS", (None, None))
    if openai_status == "success":
        st.success("Connected to OpenAI")
    elif openai_status == "error":
        st.error(openai_message)
    else:
        st.warning("Visit `Information` to connect to OpenAI")


def connect_to_weaviate(weaviate_url, weaviate_api_key, is_default_instance):
    """Try to connect to Weaviate using the URL and API key.
    Set the state variable WEAVIATE_STATUS based on the outcome.
    If the credentials are provided by the user set the 
    """
    try:
        weaviate_client = weaviate.Client(
            url=weaviate_url, 
            auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key),
        )
    except Exception as e:
        st.session_state["WEAVIATE_STATUS"] = "error", e
        return

    if weaviate_client.is_live() and weaviate_client.is_ready():
        st.session_state["WEAVIATE_CLIENT"] = weaviate_client

        if is_default_instance:
            st.session_state["WEAVIATE_DEFAULT_INSTANCE"] = True
            st.session_state["WEAVIATE_STATUS"] = "success", None
        else:
            st.session_state["WEAVIATE_DEFAULT_INSTANCE"] = False
            st.session_state["WEAVIATE_STATUS"] = "success", None
    else:
        st.session_state["WEAVIATE_STATUS"] = "error", ConnectionError("Weaviate server is not ready.")


def user_auth_weaviate():
    """Container to enter Weaviate credentials"""
    form = st.form("weaviate_auth_form")

    weaviate_url = form.text_input(
        "Weaviate Instance URL",
        help="[docs](https://weaviate.io/developers/wcs/quickstart)",
    )

    weaviate_api_key = form.text_input(
        "Weaviate API key",
        type="password",
        help="[docs](https://weaviate.io/developers/wcs/quickstart)"
    )

    if form.form_submit_button("Connect"):
        connect_to_weaviate(weaviate_url, weaviate_api_key, is_default_instance=False)


def default_auth_weaviate():
    """Connect to the default/public Weaviate instance"""
    if st.session_state.get("WEAVIATE_CLIENT"):
        return
    
    if st.session_state.get("WEAVIATE_DEFAULT_INSTANCE") is False:
        return
    
    weaviate_url = st.secrets.get("WEAVIATE_URL")
    weaviate_api_key = st.secrets.get("WEAVIATE_API_KEY")
    connect_to_weaviate(weaviate_url, weaviate_api_key, is_default_instance=True)


def weaviate_connection_status():
    """Message about the success/failure of the Weaviate connection"""
    weaviate_status, weaviate_message = st.session_state.get("WEAVIATE_STATUS", (None, None))

    if weaviate_status == "success":
        if st.session_state["WEAVIATE_DEFAULT_INSTANCE"]:
            st.success("Connected to Default Weaviate Instance")
        else:
            st.success("Connected to User Weaviate Instance")

    elif weaviate_status == "error":
        st.error(weaviate_message)

    else:
        st.warning("Visit `Information` to connect to Weaviate")