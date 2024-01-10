from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional

model_name = "text-davinci-003"
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

# Define your desired data structure.
class Event(BaseModel):
    summary: str = Field(description="title of the event", example="Meeting with John")
    location: Optional[str] = Field(description="where the event is, or the location of the event", example="800 Howard St., San Francisco, CA 94103")
    description: Optional[str] = Field(description="What the event is about", example="Discussing the new project")
    start_time: str = Field(description="The start time of the event, in datetime format", example="2021-10-10T09:00:00-07:00")
    end_time: str = Field(description="The end time of the event, in datetime format, if not specified set to an hour later than start time", example="2021-10-10T10:00:00-07:00")
    recurrence: Optional[str] = Field(description="How often is the meeting happening", example="RRULE:FREQ=DAILY;COUNT=2")
    email1: Optional[str] = Field(description="The email of another person sharing the invite", example="kadija@")
    email2: str = Field (description="Another email that is being used to share with this person", example="kadija@")
    reminders: dict = Field(description="How often you want to be reminded of the event", example="{'useDefault': False, 'overrides': [{'method': 'email', 'minutes': 24 * 60}, {'method': 'popup', 'minutes': 10}]}")

def create_json(message, json_type):
    # And a query intented to prompt a language model to populate the data structure.

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=json_type)

    prompt = PromptTemplate(
        template="Format the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(query=message)

    output = model(_input.to_string())

    parser.parse(output)

    print(output)

    return output

create_json("Create a calendar event for tomorrow at 3pm", Event)