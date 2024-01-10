from utils.file_utils import read_content
from utils.paths_utils import llm_api_token_path
from recruiter_text_replier.hugging_face import Hugging_face
from recruiter_text_replier.openai import OpenAI

class LLM_Reply_factory:
    def __init__(self, llm_name : str):
        self.llm_name : str = llm_name
    
    def create_llm(self) -> object | NotImplementedError:
        if self.llm_name == "<open_ai>":
            return OpenAI(read_content(llm_api_token_path)["api_token_openai"])
        elif self.llm_name == "<hugging_face>":
            return Hugging_face(read_content(llm_api_token_path)["api_token_hugging_face"])
        else:
            raise NotImplementedError("No LLM with that name is available.")