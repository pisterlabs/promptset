"""Natural language assistants that can be used to answer questions."""

from dataclasses import dataclass

from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate


def make_request_prompt(request: str, template: str) -> str:
    """Construct a prompt from a request and context."""
    prompt = PromptTemplate(input_variables=["input"], template=template).format(
        input=request
    )
    return prompt


@dataclass(frozen=True)
class BasicAssistant:
    """Basic memoryless assistant that can answer questions and provide explanations."""

    text_llm: BaseLLM
    """The text language model used to generate responses."""

    template: str = (
        "You are an assistant to a human, powered by a large language model.\n\n"
        + "You are designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand. You are able to update your responses based on feedback from the user, allowing you to improve your responses over a conversation.\n\n"
        + "You are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\n"
        + "Overall, you are a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether the human needs help with a specific question or just wants to have a conversation about a particular topic, you are here to assist.\n\n"
        + "User: {input}\n"
        + "Assistant:"
    )

    def post_request(self, request: str) -> str:
        """Post a request to the language model."""
        prompt = make_request_prompt(request, self.template)
        response = self.text_llm(prompt)
        return response
