from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class InputEvaluator(BaseModel):
    is_a_query: bool = Field(
        description="""Whether the user input needs to execute an SQL query or not.""",
        required=True,
    )
    include_chart: bool = Field(
        description="""Whether the user input needs to include a chart or not."""
        """This is only relevant if the argument "is_a_query" is True.""",
        required=True,
    )

    simple_answer: bool = Field(
        description="""Whether the user input needs to be a simple text response or not."""
        """This is only relevant if the argument "simple_answer" is True.""",
        required=True,        
    )



input_evaluator_parser = PydanticOutputParser(
    pydantic_object=InputEvaluator
)