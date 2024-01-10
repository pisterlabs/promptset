import openai
import streamlit as st
from streamlit_chat import message

from gooddata.agents.common import GoodDataOpenAICommon
from gooddata.agents.maql_agent import MaqlAgent
from gooddata.agents.sdk_wrapper import GoodDataSdkWrapper


class GoodDataMaqlApp:
    def __init__(self, gd_sdk: GoodDataSdkWrapper) -> None:
        self.gd_sdk = gd_sdk
        self.workspace_id = st.session_state.workspace_id
        self.agent = MaqlAgent(
            openai_model=st.session_state.openai_model,
            openai_api_key=st.session_state.openai_api_key,
            openai_organization=st.session_state.openai_organization,
            workspace_id=st.session_state.workspace_id,
        )

    def show_model_entities(self) -> None:
        catalog = self.gd_sdk.sdk.catalog_workspace_content.get_full_catalog(self.workspace_id)
        columns = st.columns(3)
        with columns[0]:
            st.write("Attributes:")
            attributes_string = ""
            i = 1
            for attribute in catalog.attributes:
                attributes_string += f"{i}. `{attribute.title}`\n"
                i += 1
            st.markdown(attributes_string)
        with columns[1]:
            st.write("Facts:")
            facts_string = ""
            i = 1
            for fact in catalog.facts:
                facts_string += f"{i}. `{fact.title}`\n"
            st.markdown(facts_string)
        with columns[2]:
            st.write("Metrics:")
            metrics_string = ""
            i = 1
            for metric in catalog.metrics:
                metrics_string += f"{i}. `{metric.title}`\n"
            st.markdown(metrics_string)

    def render(self) -> None:
        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        try:
            user_input = st.text_area("Ask for GoodData MAQL metric:")

            columns = st.columns(2)
            with columns[0]:
                if st.button("Submit Query", type="primary"):
                    output = self.agent.process(user_input)
                    st.session_state.past.append(user_input)
                    st.session_state.generated.append(output)
            with columns[1]:
                if st.button("Clear chat history", type="primary"):
                    st.session_state["generated"] = []
                    st.session_state["past"] = []

            if st.session_state["generated"]:
                for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

            if st.checkbox("Show model entities"):
                self.show_model_entities()

        except openai.error.AuthenticationError as e:
            st.write("OpenAI unknown authentication error")
            st.write(e.json_body)
            st.write(e.headers)
