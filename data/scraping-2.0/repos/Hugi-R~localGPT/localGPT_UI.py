from typing import Any, Dict, List, Union

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.schema.messages import BaseMessage
from streamlit_extras.add_vertical_space import add_vertical_space

from run_localGPT import setup_qa


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        st.session_state["response"] = ""

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        st.session_state["response"] += token
        with st.session_state["RP"].container():
            st.write(st.session_state["response"])

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 Converse with your Data')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [LocalGPT](https://github.com/PromtEngineer/localGPT) 
 
    ''')

if "QA" not in st.session_state:
    st.session_state["QA"] = setup_qa(StreamlitCallbackHandler())

st.title('LocalGPT App 💬')

prompt = st.text_input('Input your prompt here')

if "RP" not in st.session_state:
    st.session_state["RP"] = st.empty()

if "response" not in st.session_state:
    st.session_state["response"] = ""

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = st.session_state["QA"](prompt)
    answer, docs = response["result"], response["source_documents"]
    # ...and write it out to the screen
    st.write(answer)

    # With a streamlit expander  
    with st.expander('Source documents'):
        # Write out the first
        for doc in docs: 
            st.write(f"### Source Document: {doc.metadata['source'].split('/')[-1]}")
            st.write(doc.page_content) 
            st.write("--------------------------------")