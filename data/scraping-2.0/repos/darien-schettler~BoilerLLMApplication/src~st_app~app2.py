import streamlit as st
from openai.error import OpenAIError
import runhouse as rh

from src.data_manager.output_parsing import (
    wrap_text_in_html,
    llm_response_w_sources,
)
from src.model_manager.model_ecosystem import get_llm_response
from src.st_app.components.sidebar import sidebar
from src.runhouse_ops.instance_handler import get_rh_query_fn, init_rh
from src.st_app.state_utils import (
    reset_submit_state
)


def rh_ops(self):
    self.gpu = init_rh()
    if self.gpu:
        self.query_fn = get_rh_query_fn(get_llm_response, self.gpu)

def render(self):
    self.rh_ops()
    self.init_st_app()
    self.upload_file()
    self.query()
    self.submit()

BoilerLLMApp().render()