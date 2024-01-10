from langchain import FewShotPromptTemplate, LLMChain, PromptTemplate
import openai
from dotenv import load_dotenv
from os import getenv
from data_constants import (
    PROMPT_TEMPLATE,
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    PROMPT_EXAMPLES,
)
from langchain.chat_models import ChatOpenAI

load_dotenv()

openai.api_key = getenv("OPENAI_API_KEY")
openai.organization = getenv("OPENAI_ORG_ID")


class InferenceModel:
    """
    A GPT wrapper for inferring from transactions data, using model and prompt constants
    """

    def __init__(
        self,
        model_name="gpt-4",
        prompt_examples: list[dict[str, str]] = PROMPT_EXAMPLES,
    ) -> None:
        self.model_name = model_name

        prompt = FewShotPromptTemplate(
            examples=prompt_examples,
            example_prompt=PromptTemplate(
                input_variables=list(prompt_examples[0].keys()),
                template=PROMPT_TEMPLATE,
            ),
            prefix=PROMPT_PREFIX,
            suffix=PROMPT_SUFFIX,
            input_variables=["query"],
        )

        self.chain = LLMChain(
            llm=ChatOpenAI(temperature=0, model=model_name), prompt=prompt
        )

    def infer(self, transaction: str) -> str:
        return self.chain.run(transaction)
