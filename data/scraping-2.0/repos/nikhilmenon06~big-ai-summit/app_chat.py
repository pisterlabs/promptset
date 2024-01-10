import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
from utils import get_initial_message, get_chatgpt_response, update_chat
import os
from dotenv import load_dotenv
from jinja2 import Template
import openai
from dataclasses import dataclass

load_dotenv()


@dataclass
class Const:
    MODEL_NAME: str = "gpt-4"
    GUILD_RULES_NAME: str = "Assassins_Guilds_Rules_2023_v06-New"


openai.api_key = os.getenv('OPENAI_API_KEY')


# For streamlit deployment, the api key is added to streamlit-secrets in the app settings (during/after delpoyment)
# openai.api_key = st.secrets["OPENAI_API_KEY"]


def main():
    st.set_page_config(page_title="Chatbot Application", page_icon=":robot_face:", layout="centered")
    st.image('assets/big_ai.png', width=700)

    selected_page = option_menu(None, ["Editor", "Chat"], icons=['edit', 'comments'], menu_icon="bars", default_index=0,
                                orientation="horizontal", styles={"nav-link-selected": {"background-color": "#7D9338"}})

    if selected_page == "Editor":
        editor()
    elif selected_page == "Chat":
        chat()


def chat():
    print("On Chat...................\n")
    print(st.session_state.prompt)
    if 'prompt' not in st.session_state:
        st.session_state.prompt = ""

    if 'query' not in st.session_state:
        st.session_state.query = ''

    if st.session_state.bug_flag == 1:
        st.warning("Oops, your previous message was sent to the model again", icon="🤖")

    def submit():
        st.session_state.query = st.session_state.input
        st.session_state.input = ''

    model = Const.MODEL_NAME

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def display():

        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

    st.text_input("Query: ", key="input", on_change=submit)

    if 'messages' not in st.session_state:
        st.session_state['messages'] = get_initial_message()
        st.session_state['messages'] = update_chat(st.session_state['messages'], "system", st.session_state['prompt'])

    if st.session_state.query:
        with st.spinner("generating..."):
            messages = st.session_state['messages']
            messages = update_chat(messages, "user", st.session_state.query)
            # st.write("Before  making the API call")
            # st.write(messages)
            response = get_chatgpt_response(messages, model)
            messages = update_chat(messages, "assistant", response)
            st.session_state.past.append(st.session_state.query)
            st.session_state.generated.append(response)
            display()
            # st.experimental_rerun()


def editor():
    def filename_display(filename: str):
        filename, ext = os.path.splitext(filename)
        split_filename = filename.split("_")
        mod_filename = ""
        for part in split_filename:
            mod_filename = mod_filename + part.capitalize() + " "

        mod_filename = mod_filename.strip()
        return mod_filename

    def update_text_area(selected_rag_contents_pass):
        guild_rules_clean = Const.GUILD_RULES_NAME.lower().lower().replace('-', '_')
        template = Template(st.session_state["text_widget"].replace(Const.GUILD_RULES_NAME, guild_rules_clean))
        prompt_template_jinja_variable = guild_rules_clean
        dict_to_render = {prompt_template_jinja_variable: selected_rag_contents_pass}
        st.session_state.text_widget = template.render(dict_to_render)
        st.session_state.prompt = st.session_state.text_widget
        # print("Prompt session variable after callback")
        # print(st.session_state.prompt)
        # print("Text Widget session variable after callback")
        # print(st.session_state.text_widget)

    if 'generated' in st.session_state:
        st.session_state.bug_flag = 1
        # st.warning("Oops, your previous message was sent to the model again", icon = "🤖")
    else:
        st.session_state.bug_flag = 0

    if 'prompt' not in st.session_state:
        st.session_state.prompt = ""

    if 'text_widget' not in st.session_state:
        st.session_state.text_widget = ""

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to /prompts subdirectory
    prompts_dir = os.path.join(script_dir, 'prompts')

    # List all files in the /prompts directory
    files = [f for f in os.listdir(prompts_dir) if os.path.isfile(os.path.join(prompts_dir, f))]

    # Add a select box for all files
    selected_file = st.selectbox('Select a file:', files, format_func=filename_display)
    selected_file_path = os.path.join(prompts_dir, selected_file)

    with open(selected_file_path, 'r') as prompt_template_file:
        prompt_template_file_content = prompt_template_file.read()
    # template = Template(prompt_template_file_content.replace(Const.GUILD_RULES_NAME,
    #                                                          Const.GUILD_RULES_NAME.lower().lower().replace('-',
    #                                                                                                         '_')))
    # st.session_state.prompt = prompt_template_file_content

    # List all files in the /rules directory
    rules_dir = os.path.join(script_dir, 'rules')
    files = [f for f in os.listdir(rules_dir) if os.path.isfile(os.path.join(rules_dir, f))]

    with st.expander("Select data source"):
        selected_rag_file = st.selectbox('', files, format_func=filename_display, index=0)
        selected_rag_path = os.path.join(rules_dir, selected_rag_file)
        with open(selected_rag_path, "r") as file:
            selected_rag_contents = file.read()

    st.text_area(label="Write your prompt here:", height=200, key="text_widget", value=prompt_template_file_content)

    if st.button("Save Prompt", on_click=update_text_area, args=(selected_rag_contents,)):
        st.success("Prompt saved successfully!")

    st.markdown(
        'Like it? Want to talk? Get in touch! <a href="mailto:leonid.sokolov@big-picture.com">Leonid Sokolov !</a> // '
        '<a href="mailto:hello@streamlit.io">Imke Bewersdorf</a>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
