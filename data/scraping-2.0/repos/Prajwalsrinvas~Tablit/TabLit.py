import os

import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import LangChainTracer, StreamlitCallbackHandler
from langchain.llms import Clarifai
from langchain.schema.output_parser import OutputParserException
from langsmith import Client

from utils import (TTL, create_zip_file, delete_empty_subfolders, file_formats,
                   get_session_id, load_data, read_config, save_chat,
                   search_platform)

st.set_page_config(page_title="TabLit", page_icon="💊", layout="wide")

if not (
    st.session_state.get("readme_viewed") and st.session_state.get("gallery_viewed")
):
    st.toast("Please view the ReadMe and Gallery!", icon="🔥")

delete_empty_subfolders("assets")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_session_id()
assets_path = os.path.join("assets", st.session_state.session_id)
os.makedirs(assets_path, exist_ok=True)


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


title = "💊 TabLit: Chat with Tabular Data"
st.subheader(title)

with st.sidebar:
    input_data, explore_data = None, None
    crawl_web = st.toggle("Crawl web for data?", value=True)
    if not crawl_web:
        uploaded_file = st.file_uploader(
            "Upload a Data file",
            type=list(file_formats.keys()),
            help="Various File formats are Support",
            on_change=clear_submit,
        )

        if uploaded_file:
            input_data = load_data(uploaded_file)
    else:
        platforms = st.multiselect(
            "Select platform to extract data:",
            ["amazon", "walmart", "google"],
            default="amazon",
        )
        if platforms:
            search_keyword = st.text_input(
                f"Enter a keyword to search on {', '.join(platforms)}:"
            )
            if platforms and search_keyword:
                df_list = []
                for platform in platforms:
                    df = search_platform(assets_path, platform, search_keyword)

                    df_list.append(df)
                if len(df_list) == 1:
                    input_data = df_list[0]
                else:
                    input_data = df_list
    if input_data is not None:
        explore_data = st.sidebar.toggle("Explore data?", value=False)
        download_assets = st.toggle("Download assets?", value=False)

        if download_assets:
            assets = os.listdir(assets_path)
            selected_files = st.multiselect(
                "Select files to include in the zip file", assets
            )

            # Create the zip file when the user clicks a button
            if st.button("Create Zip File"):
                if selected_files:
                    create_zip_file(
                        assets_path, selected_files, st.session_state.session_id
                    )
                    st.success("Zip file created and ready for download.")
                else:
                    st.warning(
                        "Please select at least one file to include in the zip file."
                    )


client = Client(
    api_url=st.secrets.LANGCHAIN_ENDPOINT, api_key=st.secrets.LANGCHAIN_API_KEY
)
ls_tracer = LangChainTracer(project_name=st.secrets.LANGCHAIN_PROJECT, client=client)
if explore_data:
    if input_data is not None:
        if isinstance(input_data, list):
            tab_count = len(input_data)
            tabs = st.tabs(platforms)
            for tab, data in zip(tabs, input_data):
                tab.dataframe(data)
        else:
            st.dataframe(input_data)
else:
    if "messages" not in st.session_state or st.sidebar.button(
        "Clear conversation history"
    ):
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "How can I help you today?",
            }
        ]
        st.session_state["last_run"] = None

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input(placeholder="What is this data about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        llm = Clarifai(
            pat=st.secrets.CLARIFAI_PAT,
            user_id="openai",
            app_id="chat-completion",
            model_id="GPT-4",
        )

        if input_data is None:
            st.warning("Please provide input tabular data")
            st.stop()

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            input_data,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        config = read_config()
        custom_prompt = config.get("custom_prompt").format(assets_path=assets_path)
        pandas_df_agent.agent.llm_chain.prompt.template = (
            custom_prompt + pandas_df_agent.agent.llm_chain.prompt.template
        )

        col1, col2 = st.columns(2)
        with col1:
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
                try:
                    response = pandas_df_agent(
                        st.session_state.messages,
                        callbacks=[ls_tracer, st_cb],
                        include_run_info=True,
                    )
                except OutputParserException as e:
                    st.error("Please reload app due to issue in LLM output")
                    st.error(f"OutputParserException: {e}")
                st.session_state.last_run = response["__run"].run_id
                output = response["output"]
                intermediate_steps = response["intermediate_steps"]
                st.session_state.messages.append(
                    {"role": "assistant", "content": output}
                )

        st.markdown(output)
        with col2:
            if intermediate_steps:
                st.info("🤖 Internal Monologue")
                for step in intermediate_steps:
                    st.success(step[0].log)
        save_chat(assets_path, response)

    @st.cache_data(ttl=TTL, show_spinner=False)
    def get_run_url(run_id):
        return client.read_run(run_id).url

    from streamlit_feedback import streamlit_feedback

    if st.session_state.get("last_run"):
        run_url = get_run_url(st.session_state.last_run)
        st.sidebar.markdown(f"[Latest LangSmith Trace: 🛠️]({run_url})")
        feedback = streamlit_feedback(
            feedback_type="faces",
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{st.session_state.last_run}",
        )
        if feedback:
            scores = {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0}
            client.create_feedback(
                st.session_state.last_run,
                feedback["type"],
                score=scores[feedback["score"]],
                comment=feedback.get("text", None),
            )
            st.toast("Feedback recorded!", icon="📝")
