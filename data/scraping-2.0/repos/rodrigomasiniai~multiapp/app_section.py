from pathlib import Path

import streamlit as st

# DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(page_title="AI writer assistant", page_icon="img/Oxta_MLOpsFactor_logo.png",)
hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
# Design hide "made with streamlit" footer menu area
hide_streamlit_footer = """<style>#MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

from st_pages import Page, Section, add_page_title, show_pages

show_pages(
        [
        Page("app_section.py", "Home", "🏠"),
        Section(name="Doc Parser", icon="📄"),
        Page("pdf_converter.py", "PDF to txt using OCR", "📝"),
        Page("doc2pdf.py", "Docx to PDF", "📝"),
        Section(name="Doc handler with GPT", icon="📄"),
        Page("doc_gpt.py", "Doc QA", icon=":books:"),
        Page("summarizer.py", "Doc Summarization", icon="📖"),
        Section(name="Send email", icon="📤"),
        Page("email.py", "Email assistant", "✏️"),
        ]
    )

add_page_title()  # Optional method to add title and icon to current page
    # Also calls add_indentation() by default, which indents pages within a section

"## AI personal assistant with multiple tools"
import utils_api as api
import logging
import requests
import datetime

def get_openai_subscription_details(api_key, org_id=None):
    '''
    Get subscription details from OpenAI
    '''
    headers = {
        'authority': 'api.openai.com',
        'accept': '*/*',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'authorization': f'Bearer {api_key}',
        'origin': 'https://platform.openai.com',
        'referer': 'https://platform.openai.com/',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.53 Safari/537.36',

    }
    if org_id:
        headers['openai-organization'] = f'org-{org_id}'
    response = requests.get('https://api.openai.com/dashboard/billing/subscription', headers=headers)
    return response.json()


def get_openai_cost_details(api_key, start_date, end_date, org_id=None):
    '''
    Get cost details from OpenAI.
    start_date and end_date format: 2023-05-28
    '''
    headers = {
        'authority': 'api.openai.com',
        'accept': '*/*',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'authorization': f'Bearer {api_key}',
        'origin': 'https://platform.openai.com',
        'referer': 'https://platform.openai.com/',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.53 Safari/537.36',
    }
    if org_id:
        headers['openai-organization'] = f'org-{org_id}'
    params = {
        'end_date': end_date,
        'start_date': start_date
    }
    response = requests.get(
        'https://api.openai.com/dashboard/billing/usage',
        params=params,
        headers=headers
    )
    return response.json()

## helper strings
help_msg_model_temperature = "Controls how creativity in AI's response"
help_msg_model_top_p = "Prevents AI from giving certain answers that are too obvious"
help_msg_model_freq_penalty = "Encourages AI to be more diverse in its answers"
help_msg_model_presence_penalty = "Prevents AI from repeating itself too much"
help_msg_max_token = "OpenAI sets a limit on the number of tokens, or individual units of text, that each language model can generate in a single response. For example, text-davinci-003 has a limit of 4000 tokens, while other models have a limit of 2000 tokens. It's important to note that this limit is inclusive of the length of the initial prompt and all messages in the chat session. To ensure the best results, adjust the max token per response based on your specific use case and anticipated dialogue count and length in a session."
helper_app_start = "Enter an initial prompt and, optionally a follow-up message, and click on _Fetch AI Responses_ to get responses from the OpenAI models. Each time you click on _Fetch AI Responses_, the app will add the AI responses and the user messages to the chat histories of each model. You can also adjust the model parameters and keep adding follow-up messages to test the model further. To start a new test with fresh chat histories, click on _Start a new Test_. Please note the chat messages are not truncated, so you should pay attention to the total token count of your chat sessions and ensure they are within the maximum token limit of the models."
helper_app_need_api_key = "Welcome! This app allows you to test the effectiveness of your prompts using OpenAI's text models: gpt-3.5-turbo, text-davinci-003, and gpt-4 (if you have access to it). To get started, simply enter your OpenAI API Key below."
helper_api_key_prompt = "The model comparison tool works best with pay-as-you-go API keys. Free trial API keys are limited to 3 requests a minute, not enough to test your prompts. For more information on OpenAI API rate limits, check [this link](https://platform.openai.com/docs/guides/rate-limits/overview).\n\n- Don't have an API key? No worries! Create one [here](https://platform.openai.com/account/api-keys).\n- Want to upgrade your free-trial API key? Just enter your billing information [here](https://platform.openai.com/account/billing/overview)."
helper_api_key_placeholder = "Paste your OpenAI API key here (sk-...)"

def get_dates():
    today = datetime.date.today()
    beginning_of_month = datetime.date(today.year, today.month, 1)
    return today, beginning_of_month

today_date, beginning_of_month_date = get_dates()

# Handlers
def handler_verify_key():
        """Handle OpenAI key verification"""
        oai_api_key = st.session_state.open_ai_key_input
        o = api.open_ai(api_key=oai_api_key, restart_sequence='|UR|', stop_sequence='|SP|')
        try:
                # make a call to get available models
                open_ai_models = o.get_models()
                st.session_state.openai_model_params = [('gpt-3.5-turbo', 4096), ('text-davinci-003', 4000)]

                # check to see if the API key has access to gpt-4
                for m in open_ai_models['data']:
                        if m['id'] == 'gpt-4':
                                st.session_state.openai_model_params = [('gpt-4', 8000), ('gpt-3.5-turbo', 4096),
                                                                        ('text-davinci-003', 4000)]

                st.session_state.openai_models = [model_name for model_name, _ in st.session_state.openai_model_params]
                st.session_state.openai_models_str = ', '.join(st.session_state.openai_models)
                st.session_state.chat_histories = {model: [] for model in st.session_state.openai_models}
                st.session_state.total_tokens = {model: 0 for model in st.session_state.openai_models}
                st.session_state.prompt_tokens = {model: 0 for model in st.session_state.openai_models}
                st.session_state.completion_tokens = {model: 0 for model in st.session_state.openai_models}
                st.session_state.conversation_cost = {model: 0 for model in st.session_state.openai_models}

                # store OpenAI API key in session states
                st.session_state.oai_api_key = oai_api_key

                # enable the test
                st.session_state.test_disabled = False

        except Exception as e:
                with openai_key_container:
                        st.error(f"{e}")
                logging.error(f"{e}")


def handler_fetch_model_responses():
        """Fetches model responses"""

        model_config_template = {
                'max_tokens': st.session_state.model_max_tokens,
                'temperature': st.session_state.model_temperature,
                'top_p': st.session_state.model_top_p,
                'frequency_penalty': st.session_state.model_frequency_penalty,
                'presence_penalty': st.session_state.model_presence_penalty
        }

        o = api.open_ai(api_key=st.session_state.oai_api_key, restart_sequence='|UR|', stop_sequence='|SP|')
        progress = 0
        user_query_moderated = True

        init_prompt = st.session_state.init_prompt

        # Moderate prompt
        if init_prompt and init_prompt != '':
                try:
                        moderation_result = o.get_moderation(user_message=init_prompt)
                        if moderation_result['flagged'] == True:
                                user_query_moderated = False
                                flagged_categories_str = ", ".join(moderation_result['flagged_categories'])
                                with openai_key_container:
                                        st.error(
                                                f"⚠️ Your prompt has been flagged by OpenAI's content moderation endpoint due to the following categories: {flagged_categories_str}.  \n" +
                                                "In order to comply with [OpenAI's usage policy](https://openai.com/policies/usage-policies), we cannot send this prompt to the models. Please modify your prompt and try again.")
                except Exception as e:
                        logging.error(f"{e}")
                        with openai_key_container:
                                st.error(f"{e}")

        # Moderate follow-up message
        if "user_msg" in st.session_state and st.session_state.user_msg != '':
                try:
                        moderation_result = o.get_moderation(user_message=st.session_state.user_msg)
                        if moderation_result['flagged'] == True:
                                user_query_moderated = False
                                flagged_categories_str = ", ".join(moderation_result['flagged_categories'])
                                with openai_key_container:
                                        st.error(
                                                f"⚠️ Your most recent follow up message has been flagged by OpenAI's content moderation endpoint due to the following categories: {flagged_categories_str}.  \n" +
                                                "In order to comply with [OpenAI's usage policy](https://openai.com/policies/usage-policies), we cannot send this message to the models. Please modify your message and try again.")
                except Exception as e:
                        logging.error(f"{e}")
                        with openai_key_container:
                                st.error(f"{e}")

        if init_prompt and init_prompt != '' and user_query_moderated == True:
                for index, m in enumerate(st.session_state.openai_models):
                        progress_bar_container.progress(progress, text=f"Getting {m} responses")
                        if "user_msg" in st.session_state and st.session_state.user_msg != '':
                                st.session_state.chat_histories[m].append(
                                        {'role': 'user', 'message': st.session_state.user_msg,
                                         'created_date': api.get_current_time()})

                        try:
                                b_r = o.get_ai_response(
                                        model_config_dict={**model_config_template, 'model': m},
                                        init_prompt_msg=init_prompt,
                                        messages=st.session_state.chat_histories[m]
                                )

                                st.session_state.chat_histories[m].append(b_r['messages'][-1])
                                st.session_state.total_tokens[m] = b_r['total_tokens']
                                st.session_state.prompt_tokens[m] = b_r['prompt_tokens']
                                st.session_state.completion_tokens[m] = b_r['completion_tokens']

                                if m == 'gpt-4':
                                        # $0.03 / 1K prompt tokens + $0.06 / 1K completion tokens
                                        st.session_state.conversation_cost[m] = 0.03 * st.session_state.prompt_tokens[
                                                m] / 1000 + 0.06 * st.session_state.completion_tokens[m] / 1000
                                elif m == 'gpt-3.5-turbo':
                                        # 0.002 / 1K total tokens
                                        st.session_state.conversation_cost[m] = 0.002 * st.session_state.total_tokens[
                                                m] / 1000
                                elif m == 'text-davinci-003':
                                        # 0.02 / 1K total tokens
                                        st.session_state.conversation_cost[m] = 0.02 * st.session_state.total_tokens[
                                                m] / 1000

                                # update the progress bar
                                progress = (index + 1) / len(st.session_state.openai_models)
                                progress_bar_container.progress(progress, text=f"Getting {m} responses")

                        except o.OpenAIError as e:
                                logging.error(f"{e}")
                                with openai_key_container:
                                        if e.error_type == "RateLimitError" and str(
                                                e) == "OpenAI API Error: You exceeded your current quota, please check your plan and billing details.":
                                                st.error(
                                                        f"{e}  \n  \n**Friendly reminder:** If you are using a free-trial OpenAI API key, this error is caused by the limited rate limits associated with the key. To optimize your experience, we recommend upgrading to the pay-as-you-go OpenAI plan.")
                                        else:
                                                st.error(f"{e}")

                        except Exception as e:
                                with openai_key_container:
                                        st.error(f"{e}")
                                logging.error(f"{e}")

        progress_bar_container.empty()


def handler_start_new_test():
        """Start new test"""
        st.session_state.chat_histories = {model: [] for model in st.session_state.openai_models}
        st.session_state.total_tokens = {model: 0 for model in st.session_state.openai_models}
        st.session_state.prompt_tokens = {model: 0 for model in st.session_state.openai_models}
        st.session_state.completion_tokens = {model: 0 for model in st.session_state.openai_models}
        st.session_state.conversation_cost = {model: 0 for model in st.session_state.openai_models}

def ui_introduction():
        col1, col2 = st.columns([6, 4])
        col1.text_input(label="Enter OpenAI API Key", key="open_ai_key_input", type="password",
                        autocomplete="current-password", on_change=handler_verify_key,
                        placeholder=helper_api_key_placeholder, help=helper_api_key_prompt)


def ui_test_result(progress_bar_container):
        progress_bar_container.empty()

        if "openai_models" in st.session_state:
                columns = st.columns(len(st.session_state.openai_models))

                for index, model_name in enumerate(st.session_state.openai_models):
                        if len(st.session_state.chat_histories[model_name]) > 0:
                                with columns[index]:
                                        st.write(f'_Conversation with {model_name}_')
                                        st.write(f'Total tokens: {st.session_state.total_tokens[model_name]}')
                                        st.write(f'Prompt tokens: {st.session_state.prompt_tokens[model_name]}')
                                        st.write(f'Completion tokens: {st.session_state.completion_tokens[model_name]}')
                                        st.write(f'Total cost: ${st.session_state.conversation_cost[model_name]}')
                                        st.write("---")
                                        for message in st.session_state.chat_histories[model_name]:
                                                if message['role'] == 'user':
                                                        st.markdown(f"**User:**  \n{message['message']}")
                                                else:
                                                        st.markdown(f"**Model:**  \n{message['message']}")


def ui_ctas():
        col1, col2, = st.columns(2)
        if "chat_histories" in st.session_state and len(
                st.session_state.chat_histories[st.session_state.openai_models[-1]]) > 0:
                st.button(label="Start a new chat", on_click=handler_start_new_test)
                st.write("---")
        with col1:
                st.text_area(
                        label="Initial prompt",
                        height=100,
                        max_chars=2000,
                        key="init_prompt",
                        disabled=st.session_state.test_disabled
                )
                st.number_input(label="Response Token Limit", key='model_max_tokens', min_value=0, max_value=1000,
                        value=300, step=50, help=help_msg_max_token, disabled=st.session_state.test_disabled)
        st.slider(label="Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.7,
                  key='model_temperature', help=help_msg_model_temperature,
                  disabled=st.session_state.test_disabled)
        st.slider(label="Top P", min_value=0.0, max_value=1.0, step=0.1, value=1.0, key='model_top_p',
                  help=help_msg_model_top_p, disabled=st.session_state.test_disabled)
        st.slider(label="Frequency penalty", min_value=0.0, max_value=1.0, step=0.1, value=0.0,
                  key='model_frequency_penalty', help=help_msg_model_freq_penalty,
                  disabled=st.session_state.test_disabled)
        st.slider(label="Presence penalty", min_value=0.0, max_value=1.0, step=0.1, value=0.0,
                  key='model_presence_penalty', help=help_msg_model_presence_penalty,
                  disabled=st.session_state.test_disabled)
        st.button(label="Fetch AI Responses", on_click=handler_fetch_model_responses,
                  disabled=st.session_state.test_disabled)
        with col2:
                st.text_area(
                        label="Follow up message",
                        height=100,
                        max_chars=1000,
                        key="user_msg",
                        disabled=st.session_state.test_disabled
                )


## initialize states
if "test_disabled" not in st.session_state:
        st.session_state.test_disabled = True

## UI
openai_key_container = st.container()

st.title('OpenAI GPT Model Comparison Tool')
if "oai_api_key" not in st.session_state:
        st.write(helper_app_need_api_key)
        ui_introduction()
        with openai_key_container:
                st.empty()

        st.write("---")
        ui_ctas()

else:
        if "openai_models_str" in st.session_state:
                st.write(f"Write your prompts on following OpenAI models: {st.session_state.openai_models_str}")

        st.write(helper_app_start)

        ui_ctas()

        st.write("---")

        with st.expander(label="Initial Prompt", expanded=True):
                st.write(st.session_state.init_prompt)

        with openai_key_container:
                st.empty()

        st.write("##### Test chat sessions")

        progress_bar_container = st.empty()
        ui_test_result(progress_bar_container)
        st.subheader("Subscription details:")
        st.write(
            f'Current subscription details: {get_openai_subscription_details(api_key=st.session_state.oai_api_key)}')
        st.write(
            f'Current costs: {get_openai_cost_details(api_key=st.session_state.oai_api_key, start_date=beginning_of_month_date, end_date=today_date)}')

