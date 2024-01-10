import json
import tiktoken

from typing import List
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.tools import format_tool_to_openai_function, tool

from GPTagger.logger import log2cons
from GPTagger.constants import model2ctxlen


# The schema seems to be very important
class Extractions(BaseModel):
    texts: List[str] = Field(
        description=(
            "List of strings extracted from the text according to the instructions"
        )
    )


@tool(args_schema=Extractions)
def process_extractions(texts: List[str]):
    """Process the list of extracted text"""
    return texts


class Textractor:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo-0613",
        num_of_calls: int = 1,
        use_tool: bool = True,
        max_new_tokens: int = 256,
    ):
        """Textractor request gpt to get extractions

        Args:
            model (str, optional): the used GPT model. Defaults to "gpt-3.5-turbo-0613".
            num_of_calls (int, optional): number of calls. Defaults to 1.
            use_tool (bool, optional): use functional call feature or not. Defaults to True.
            max_new_tokens (int, optional): max length of generated token. Defaults to 256.

        """
        if model not in model2ctxlen:
            raise ValueError(f"Unsupported model name {model}")
        # model setup
        self.use_tool = use_tool
        self.num_of_calls = num_of_calls
        self.limit = model2ctxlen[model]
        self.model = ChatOpenAI(model=model, max_tokens=max_new_tokens)
        # estimate token usage
        self.tkctr = 0
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _request(self, prompt: str) -> List[str]:
        """request GPT with a prompt and get a list of extractions

        Args:
            prompt (str): the prompt

        Returns:
            List[str]: list of extractions
        """
        if self.use_tool:
            # The function_call param is very important to restrict the model to only call this function
            msg = self.model.predict_messages(
                [HumanMessage(content=prompt)],
                functions=[format_tool_to_openai_function(process_extractions)],
                function_call={"name": "process_extractions"},
            )
            function_call = msg.additional_kwargs["function_call"]
            texts = json.loads(function_call["arguments"])["texts"]
            if isinstance(texts, str):
                texts = texts.split("\n")
        else:
            # Either the content is json or it should be multiple line content
            msg = self.model.predict_messages([HumanMessage(content=prompt)])
            try:
                texts = json.loads(msg.content)["texts"]
                if isinstance(texts, str):
                    texts = texts.split("\n")
            except:
                texts = msg.content.split("\n")

        return texts

    def request(self, prompt: str) -> List[str]:
        """request GPT, call multiple times based on `nr_calls`

        Args:
            prompt (str): the prompt

        Returns:
            List[str]: list of extractions
        """
        tks = self.encoder.encode(prompt)
        # Reach limit of llm
        if len(tks) > self.limit:
            prompt = self.encoder.decode(tks[: self.limit - 10]) + '\n"""'
            log2cons.warn(
                f"Current prompt has length {len(tks)}, exceed the limit of"
                f" {self.limit}"
            )

        extractions = []
        with get_openai_callback() as cb:
            for _ in range(self.num_of_calls):
                try:
                    extractions.extend(self._request(prompt))
                except Exception as e:
                    log2cons.exception("Got Extractor Error")
            self.tkctr += cb.total_tokens

        # remove duplications
        extractions = list(set(extractions))

        return extractions

    def __call__(self, text: str, template: PromptTemplate) -> List[str]:
        """request gpt with prompt template and text

        Args:
            text (str): text where extraction happens
            template (PromptTemplate): prompt template with {text} placeholder

        Returns:
            List[str]: list of extracted strings
        """
        prompt = template.format(text=text)
        extractions = self.request(prompt)

        return extractions
