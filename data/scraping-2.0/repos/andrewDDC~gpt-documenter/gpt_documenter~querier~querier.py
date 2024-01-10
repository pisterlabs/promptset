import tiktoken
from langchain import (
    BasePromptTemplate,
    LLMChain,
)
from langchain.chat_models import ChatOpenAI

from gpt_documenter.documenter import templates


class Querier:
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            temperature=0.0,
            openai_api_key=self.openai_api_key,
        ) if openai_api_key is not None else None
        self.prompts_sent = []
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def send_query(
            self,
            prompt: BasePromptTemplate,
            initial: str = "",
            **kwargs,
    ):

        query_sent_text = prompt.format_prompt(**kwargs).dict()["text"]
        query_save_object = {"prompt": query_sent_text}

        chain = LLMChain(llm=self.llm, prompt=prompt)

        results = initial + chain.run(**kwargs)
        query_save_object["response"] = results

        full_text = results + query_sent_text
        token_usage = self.get_string_tokens(full_text)
        query_save_object["token_usage"] = token_usage
        self.prompts_sent.append(query_save_object)

        return results

    def calculate_function_tokens(self, function_text: str, functions_used: int):
        json_template = templates.summary_json_template()
        tokens_json_response_template = self.get_string_tokens(json_template)
        avg_function_doc_tokens = 310

        function_tokens = self.get_string_tokens(function_text)
        function_tokens += tokens_json_response_template
        # base function's template
        if functions_used == 0:
            base_template = templates.doc_base_function_template().template
            base_template_tokens = self.get_string_tokens(base_template)
            function_tokens += base_template_tokens
        else:
            composed_template = templates.doc_function_template().template
            composed_template_tokens = self.get_string_tokens(composed_template)
            function_tokens += composed_template_tokens
            function_tokens += functions_used * avg_function_doc_tokens

        return function_tokens

    def get_string_tokens(self, text: str):
        return len(self.encoder.encode(text))

