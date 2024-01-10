from app.task_resolver.engine import StepResolver

from typing import List, Any
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from langchain.chat_models import ChatOpenAI
from app.utils import chain_verbose
from langchain.llms import OpenAI
from app.model import Message

# BOOKING_CONFIRMATION_STEP:
# HOUSE_SELECTION_STEP:

template="""Given a conversation between a user and an assistant about booking a house for short-term stay. \
Your job is to decide if the conversation came to an end already.

A conversation came to an end in the following cases:
1. After the user gets a confirmation from the assistant that the reservation in booked for some time and that an email will be sent to the email provided by her.
2. When the user decides not to book a reservation after the assistant asked to confirm the booking.
3. When there are no properties available for the user's booking requirements and the user does not want to pick other dates for the reservation.
4. When the user is making a reservation but suddenly wants to perform some other task not related with making reservations.
5. When the user explicitly ask to end the conversation.

On every other case the conversation is still active.

{format_instructions}

Current conversation: 
{chat_history}"""

response_schemas = [
    ResponseSchema(name="conversation_finished", type="bool", description="true if the conversation between the user and the assistant came to an end, otherwise false."),
    ResponseSchema(name="text", description="Response to the user."),
]

class ExitTaskChain:

    def __init__(self):

        llm = OpenAI(model_name="text-davinci-003", temperature=0.)
        
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = self.output_parser.get_format_instructions()

        prompt_template = PromptTemplate(
            input_variables=["chat_history"], 
            partial_variables={"format_instructions": format_instructions},
            template=template
        )

        self.chain = LLMChain(llm=llm, 
                              prompt=prompt_template, 
                              verbose=chain_verbose,
                              output_key="result")

    def __call__(self, chat_history, current_task):

        info = self.chain({"chat_history": chat_history})
        return self.output_parser.parse(info["result"])


class ExitTaskResolver(StepResolver):

    exit_task_chain: ExitTaskChain = ExitTaskChain()

    def run(self, messages: List[Message], previous_steps_data: dict, step_chat_history: List[Message] = None) -> Message:
        chat_history = self.build_chat_history(messages)

        # current_task = step_data["current_task_name"]
        exit_result = self.exit_task_chain(chat_history, "")

        self.data["conversation_finished"] = exit_result["conversation_finished"]
        if ("conversation_finished" in exit_result and  
            exit_result["conversation_finished"] != "" and
            exit_result["conversation_finished"] == True):
            return None
        
    def is_done(self):
        # Force to execute this step every time.
        return (
            "conversation_finished" in self.data and  
            self.data["conversation_finished"] != "" and 
            self.data["conversation_finished"] is not None
        )
            