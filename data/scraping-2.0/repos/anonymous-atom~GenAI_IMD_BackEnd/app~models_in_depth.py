from __future__ import annotations
import os
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from typing import Any
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import re
import json
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

OPENAI_API_KEY = "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"
prompt_file = "/home/karun/Downloads/GenAI_IMD_BackEnd_Abhi/GenAI_IMD_BackEnd/Prompt_template.txt"


class TopicDescGen(LLMChain):
    """LLM Chain specifically for generating multi-paragraph rich text topic description using emojis."""

    @classmethod
    def from_llm(
            cls, llm: BaseLanguageModel, prompt: str, **kwargs: Any
    ) -> TopicDescGen:
        """Load ProductDescGen Chain from LLM."""
        return cls(llm=llm, prompt=prompt, **kwargs)


def topic_desc_generator(topic_name, keywords):
    with open(prompt_file, "r") as file:
        prompt_template = file.read()

    PROMPT = PromptTemplate(
        input_variables=["topic_name", "keywords"], template=prompt_template
    )
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY,
    )

    TopicDescGen_chain = TopicDescGen.from_llm(llm=llm, prompt=PROMPT)
    TopicDescGen_query = TopicDescGen_chain.apply_and_parse(
        [{"topic_name": topic_name, "keywords": keywords}]
    )

    return TopicDescGen_query[0]["text"]


def Question_bank(text):

    response_schemas = [
        ResponseSchema(name="question", description="A multiple choice question generated from input text snippet."),
        ResponseSchema(name="options", description="Possible choices for the multiple choice question."),
        ResponseSchema(name="answer", description="Correct answer for the question.")
    ]

    # The parser that will look for the LLM output in my schema and return it back to me
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # The format instructions that LangChain makes. Let's look at them
    format_instructions = output_parser.get_format_instructions()

    PROMPT = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                """Given a text input, generate multiple choice questions from it along with the correct answer. \n{format_instructions}\n{user_prompt}""")],
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions}
    )

    chat_model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY,
    )

    user_query = PROMPT.format_prompt(user_prompt=text)
    user_query_output = chat_model(user_query.to_messages())
    return user_query_output.content



# with gr.Blocks() as demo:
#     gr.HTML("""<h1>Welcome to my class students</h1>""")
#     gr.Markdown(
#         "Want to learn about a topic in details in one search???<br>"
#         "Provide topic name and keywords you want to lear about. Click on 'Generate Description' button and multi-paragraph rich text product description will be genrated instantly."
#     )
#
#     with gr.Tab("Expalin the topic!"):
#         topic_name = gr.Textbox(
#             label="Topic Name",
#             placeholder="CNN",
#         )
#         keywords = gr.Textbox(
#             label="Keywords (separated by commas)",
#             placeholder="Convolutionaal operators, kernels, Neural networks",
#         )
#         detailed_description = gr.Textbox(label="In-depth learn")
#         click_button = gr.Button(value="Teach me master!")
#         click_button.click(
#             topic_desc_generator, [topic_name, keywords], detailed_description
#         )
# demo.launch(share=True)
