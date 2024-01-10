import json

from langchain import PromptTemplate
from langchain.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

from LangChainPipeline.PydanticClasses.TemporalSegments import TemporalSegments


PREFIX_TEMPORAL_POSITION_PROMPT= """
You are a video editor's assistant who is trying to understand natural language temporal reference in the video. You will do it step-by-step.

First step: Identify the type of temporal reference based on the user's command.
1. Timecode: a specific time in the video
2. Time range: a range of time in the video
3. More high level temporal reference: a reference to a generic event in the video (introduction, ending, etc.)

Second step: Identify the timecode or time range with additional context.
Note 1: If the temporal reference is just a timecode, output any 10 second interval containing the timecode.
Note 2: If there are more than one segment of video that matches the temporal reference, output all of them in a list.

{format_instructions}
"""

EXAMPLE_PROMPT = """
Command: {command}
Context: {context}
Response: {response}
"""

SUFFIX_TEMPORAL_POSITION_PROMPT = """
Command: {command}
Context: {context}
Response:
"""

def get_examples():
    context1 = ["A video is 10 minutes long.",
        "You are at 05:23",
        "The original command was: 0:07 add a text saying: introduction",
    ]
    command1 = ["0:07"]
    response1 = TemporalSegments.get_instance(start="00:00:04", finish="00:00:09")

    
    context2 = [
        "A video is 45:13 long.",
        "You are at 00:00:00",
        "The original command was: in the intro, blur the background",
    ]
    command2 = ["intro"]
    response2 = TemporalSegments.get_instance(start="00:00:00", finish="00:00:30")

    context3 = [
        "A video is 00:20:13 long.",
        "You are at 5 minutes",
        "The original command was: Add an image of a cat between 5:10 and 5:20",
    ]
    command3 = ["between 5:10 and 5:20"]
    response3 = TemporalSegments.get_instance(start="00:05:10", finish="00:05:20")

    context4 = [
        "A video is 5 minutes long.", 
        "You are at 32 seconds",
        "The original command was: At current time add a text saying: The video is about...",
    ]
    command4 = ["at current time"]
    response4 = TemporalSegments.get_instance(start="00:00:32", finish="00:00:42")


    context5 = [
        "The video is an hour long",
        "You are at 10:00",
        "The original command was: After 30 seconds zoom out to show the whole room",
    ]
    command5 = ["after 30 seconds"]
    response5 = TemporalSegments.get_instance(start="00:10:30", finish="00:10:40")

    examples = []
    examples.append({
        "context": json.dumps(context1),
        "command": json.dumps(command1),
        "response": response1.model_dump_json(),
    })
    examples.append({
        "context": json.dumps(context2),
        "command": json.dumps(command2),
        "response": response2.model_dump_json(),
    })
    examples.append({
        "context": json.dumps(context3),
        "command": json.dumps(command3),
        "response": response3.model_dump_json(),
    })
    examples.append({
        "context": json.dumps(context4),
        "command": json.dumps(command4),
        "response": response4.model_dump_json(),
    })
    examples.append({
        "context": json.dumps(context5),
        "command": json.dumps(command5),
        "response": response5.model_dump_json(),
    })
    return examples

def get_temporal_position_prompt_llm(partial_variables={}, examples = []):
    example_prompt_template = PromptTemplate(
        input_variables=["context", "command", "response"],
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
        prefix=PREFIX_TEMPORAL_POSITION_PROMPT,
        suffix=SUFFIX_TEMPORAL_POSITION_PROMPT,
        input_variables=["context", "command"],
        partial_variables=partial_variables,
    )


def get_temporal_position_prompt_chat(partial_variables={}):
    example_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "Command: {command}\nContext: {context}"),
            ("ai", "{response}"),
        ]
    )
    few_shot_prompt_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt_template,
        examples=get_examples(),
    )

    system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(
            input_variables=[],    
            template=PREFIX_TEMPORAL_POSITION_PROMPT,
            partial_variables=partial_variables,
        )
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            few_shot_prompt_template,
            ("human", "Command: {command}\nContext: {context}"),
        ]
    )

    return final_prompt