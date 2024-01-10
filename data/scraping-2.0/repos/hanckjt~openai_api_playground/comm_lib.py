import streamlit as st
import openai
import requests


def test_speed_with_system_proxy(url, timeout=5):
    return requests.get(url, timeout=timeout).elapsed.total_seconds() * 1000.0


@st.cache_resource
def get_openai_client():
    client = openai.OpenAI(base_url=get_app_setting('base_url', ''), api_key=get_app_setting('api_key', ''))
    return client


def set_openai_client(base_url, api_key):
    set_app_setting('base_url', base_url)
    set_app_setting('api_key', api_key)
    get_openai_client.clear()


def openai_completion(prompt, model, max_tokens=512, temperature=0.7, top_p=1, n=1, stream=False, logprobs=None, stop=None, presence_penalty=0, frequency_penalty=0, best_of=1, echo=False):
    return get_openai_client().completions.create(prompt=prompt, model=model)


def openai_chat_completion(messages, model, stream=True, max_tokens=512, temperature=0.7, top_p=1.0, timeout=10):
    response = get_openai_client().chat.completions.create(messages=messages, model=model, max_tokens=max_tokens, stream=stream, temperature=temperature, top_p=top_p, timeout=timeout)

    return response


def get_session_state(name, default=None):
    if name not in st.session_state:
        st.session_state[name] = default

    return st.session_state[name]


def set_session_state(name, value):
    st.session_state[name] = value


def get_app_setting(name, default=None):
    return get_session_state('settings', {}).get(name, default)


def set_app_setting(name, value):
    get_session_state('settings', {})[name] = value


def clear_app_setting():
    set_session_state('settings', {})


def get_chat_history():
    return get_session_state('chat_history', [])


def add_chat_history(role, content):
    get_chat_history().append({'role': role, 'content': content})


def reset_chat_history():
    set_session_state('chat_history', [])
    add_chat_history('system', get_app_setting('system_prompt', 'You are a hellpful assistant.'))


def test():
    st.write('test')
    reset_chat_history()
    add_chat_history('user', 'test')
    st.write(get_chat_history())

    set_app_setting('base_url', 'my_base_url')
    st.write(get_app_setting('base_url'))


if __name__ == "__main__":
    test()
