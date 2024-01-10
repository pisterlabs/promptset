import json
from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

from LangChainPipeline.PydanticClasses.Rectangle import Rectangle


PREFIX_SPATIAL_POSITION_PROMPT= """
You are a video editor's assistant who is trying to understand editor's natural language description of the spatial location within the frame. The description is based on the rectangle that is already present in the frame. You will have to refine its location and resize (if necessary) based on the command.
You will be given the initial location of the rectangle in the frame: x, y, width, height, where (x, y) are coordinates of the top-left corner, and (width, height) are just width and height. Also, you will be given a command that describes the desired spatial location of the rectangle in the frame, the original context of the command, and the boundaries of the frame (e.g. width=1280, height=720)

You will do it step-by-step.
1. Refine the location of the rectangle (x, y coorindates) based on the command, original context of the command, and boundaries of the frame (make sure not to exceed the boundaries);
2. Resize the rectangle (width, height) based on the command, original context of the command, and boundaries of the frame (make sure not to exceed the boundaries);

Perform each step one-by-one and output the final location of the rectangle in the frame in appropriate format.

{format_instructions}
"""

EXAMPLE_PROMPT = """
Command: {command}
Context: {context}
Rectangle: {rectangle}
Response: {response}
"""

SUFFIX_SPATIAL_POSITION_PROMPT = """
Command: {command}
Context: {context}
Rectangle: {rectangle}
Response:
"""


def get_examples():
    #example1
    context1 = [
        "Frame Size: height: 100, width: 100",
        "The original command was: Put a rectangle in the top left corner."
    ]
    rectangle1 = Rectangle.get_instance(
        x=0,
        y=0,
        width=100,
        height=100,
        rotation=0,
    )
    command1 = ["top left corner"]
    response1 = Rectangle.get_instance(
        x=0,
        y=0,
        width=40,
        height=40,
        rotation=0,
    )
    #example2
    context2 = [
        "Frame Size: height: 480, width: 854",
        "The original command was: Whenever laptop is mentioned, put a textbox on the right side of the frame."
    ]
    rectangle2 = Rectangle.get_instance(
        x=150,
        y=300,
        width=300,
        height=100,
        rotation=0,
    )
    command2 = ["right side of the frame"]
    response2 = Rectangle.get_instance(
        x=554,
        y=300,
        width=300,
        height=100,
        rotation=0,
    )
    #example3
    context3 = [
        "Frame Size: height: 200, width: 200",
        "The original command was: Zoom should start at the bottom center."
    ]
    rectangle3 = Rectangle.get_instance(
        x=50,
        y=0,
        width=100,
        height=50,
        rotation=0,
    )
    command3 = ["bottom center"]
    response3 = Rectangle.get_instance(
        x=50,
        y=150,
        width=100,
        height=50,
        rotation=0,
    )
    #example4
    context4 = [
        "Frame Size: height: 500, width: 1000",
        "The original command was: Put the textbox with greeting text at the title-like position."
    ]
    rectangle4 = Rectangle.get_instance(
        x=300,
        y=190,
        width=100,
        height=100,
        rotation=0,
    )
    command4 = ["title-like"]
    response4 = Rectangle.get_instance(
        x=250,
        y=0,
        width=500,
        height=100,
        rotation=0,
    )
    examples = []
    examples.append({
        "context": json.dumps(context1),
        "rectangle": rectangle1.model_dump_json(),
        "command": json.dumps(command1),
        "response": response1.model_dump_json(),
    })
    examples.append({
        "context": json.dumps(context2),
        "rectangle": rectangle2.model_dump_json(),
        "command": json.dumps(command2),
        "response": response2.model_dump_json(),
    })
    examples.append({
        "context": json.dumps(context3),
        "rectangle": rectangle3.model_dump_json(),
        "command": json.dumps(command3),
        "response": response3.model_dump_json(),
    })
    examples.append({
        "context": json.dumps(context4),
        "rectangle": rectangle4.model_dump_json(),
        "command": json.dumps(command4),
        "response": response4.model_dump_json(),
    })
    return examples

def get_spatial_position_prompt_llm(partial_variables={}, examples = []):
    example_prompt_template = PromptTemplate(
        input_variables=["context", "rectangle", "command", "response"],
        template=EXAMPLE_PROMPT,
    )

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt_template,
        max_length=300,
    )

    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt_template,
        prefix=PREFIX_SPATIAL_POSITION_PROMPT,
        suffix=SUFFIX_SPATIAL_POSITION_PROMPT,
        input_variables=["context", "rectangle", "command"],
        partial_variables=partial_variables,
    )


def get_spatial_position_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Command: {command}\nContext: {context}\nRectangle: {rectangle}\n"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_SPATIAL_POSITION_PROMPT,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "Command: {command}\nContext: {context}\nRectangle: {rectangle}\n"),
        ]
    )

    return final_prompt