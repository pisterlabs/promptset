from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from src.langchain import openai_llm


def handle_llm_call(
    prompt: str, model: BaseModel, template: str, signature=None
) -> dict:
    parser = PydanticOutputParser(pydantic_object=model)

    template = PromptTemplate(
        template="{template} {format_instructions} {prompt} {signature}",
        input_variables=["format_instructions", "prompt"],
        partial_variable={
            "format_instructions": parser.get_format_instructions(),
        },
    )

    create_event_chain = LLMChain(
        llm=openai_llm, prompt=template, output_key="output"
    )

    respponse = create_event_chain(
        {
            "prompt": prompt,
            "format_instructions": parser.get_format_instructions(),
            "template": template,
            "signature": signature,
        }
    )
    parsed_event = {}
    try:
        parsed_event = parser.parse(respponse["output"])
        return parsed_event.model_dump()
    except ValueError as e:
        print(e)
        parsed_event = {"error": "Unable to parse the output"}
