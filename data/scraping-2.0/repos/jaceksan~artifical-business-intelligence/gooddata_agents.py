import argparse
import os

import openai
import streamlit as st
from dotenv import load_dotenv
from openapi_parser import parse
from openapi_parser.specification import Specification

from gooddata.agents.sdk_wrapper import GoodDataSdkWrapper
from gooddata.tools import get_name_for_id
from streamlit_apps.any_to_star import GoodDataAnyToStarApp
from streamlit_apps.api_executor import GoodDataApiExecutorApp
from streamlit_apps.chat import GoodDataChatApp
from streamlit_apps.constants import GoodDataAgent
from streamlit_apps.explain_report import GoodDataExplainReportApp
from streamlit_apps.maql import GoodDataMaqlApp
from streamlit_apps.report_executor import GoodDataReportExecutorApp

# Workaround - when we utilize "key" property in multiselect/selectbox,
#   a warning is produced if we reset the default value in a custom way
st.elements.utils._shown_default_value_warning = True


class GoodDataAgentsDemo:
    def __init__(self) -> None:
        self.args = self.parse_arguments()
        load_dotenv()
        self.gd_sdk = GoodDataSdkWrapper()
        st.set_page_config(layout="wide", page_icon="favicon.ico", page_title="Talk to GoodData")
        self.render_workspace_picker()

    # It must be set here globally
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(
            conflict_handler="resolve",
            description="Talk to GoodData",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        return parser.parse_args()

    @staticmethod
    def render_openai_key_setting():
        if "openai_api_key" not in st.session_state:
            st.session_state["openai_api_key"] = ""
        if "openai_organization" not in st.session_state:
            st.session_state["openai_organization"] = ""
        if not st.session_state.openai_api_key:
            if os.getenv("OPENAI_API_KEY"):
                st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")
                st.session_state.openai_organization = os.getenv("OPENAI_ORGANIZATION")
            else:
                token = st.text_input("OpenAI token: ", type="password")
                organization = st.text_input("OpenAI organization(optional): ")
                if st.button("Save"):
                    st.session_state.openai_api_key = token
                    st.session_state.openai_organization = organization

    def render_workspace_picker(self):
        workspaces = self.gd_sdk.sdk.catalog_workspace.list_workspaces()
        st.sidebar.selectbox(
            label="Workspaces:",
            options=[w.id for w in workspaces],
            format_func=lambda x: get_name_for_id(workspaces, x),
            key="workspace_id",
        )

    @staticmethod
    def render_agent_picker():
        st.sidebar.selectbox(
            label="Agent:",
            options=[x.name for x in list(GoodDataAgent)],
            format_func=lambda x: GoodDataAgent[x].value,
            key="agent",
        )

    @staticmethod
    def render_openai_models_picker():
        default_model = "gpt-3.5-turbo-0613"
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = default_model
        models = get_supported_models()
        st.sidebar.selectbox(
            label="OpenAI model:", options=models, key="openai_model", index=models.index(default_model)
        )

    def main(self):
        # If OPENAI credentials are not set as env variables,
        # render corresponding input fields so users can set them manually
        self.render_openai_key_setting()
        self.render_agent_picker()
        if st.session_state.openai_api_key:
            self.render_openai_models_picker()

        selected_agent = GoodDataAgent[st.session_state.get("agent")]

        if st.session_state.openai_api_key:
            if selected_agent == GoodDataAgent.CHAT:
                GoodDataChatApp().render()
            elif selected_agent == GoodDataAgent.EXPLAIN_DATA:
                GoodDataExplainReportApp(self.gd_sdk).render()
            elif selected_agent == GoodDataAgent.ANY_TO_STAR:
                GoodDataAnyToStarApp(self.gd_sdk).render()
            elif selected_agent == GoodDataAgent.REPORT_EXECUTOR:
                GoodDataReportExecutorApp(self.gd_sdk).render()
            elif selected_agent == GoodDataAgent.API_EXECUTOR:
                GoodDataApiExecutorApp().render()
            elif selected_agent == GoodDataAgent.MAQL_GENERATOR:
                GoodDataMaqlApp(self.gd_sdk).render()
            else:
                raise Exception(f"Unsupported Agent: {selected_agent=}")


@st.cache_data
def get_supported_models() -> list[str]:
    openai.api_key = st.session_state.openai_api_key
    openai.organization = st.session_state.openai_organization
    return [m["id"] for m in openai.Model.list()["data"]]


@st.cache_data
def get_api_spec() -> Specification:
    return parse("gooddata/open-api-spec.json", strict_enum=False)


if __name__ == "__main__":
    GoodDataAgentsDemo().main()
