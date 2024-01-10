import logging
from typing import Dict, List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseMessage, SystemMessage
from langchain.schema.language_model import LanguageModelInput

from jarvis.gadgets import GadgetMetadata, StopAction, StopGadget
from jarvis.gadgets.clients import HAAPIClient, HAHttpRequestAction
from jarvis.utils import log

from .interfaces import ProcessingModule

_system_msg_template = """\
You are an {name} agent. You help the user by carrying out tasks for them or providing them with \
information. To help your user you will be given several `gadgets` which you can interact with to accomplish a users \
request. A history of all gadget interactions will be stored. Follow the instructions and documentation provided by \
each tool for effective use.

gadgets: ```{gadgets}```

Determine the next action that should be executed utilizing the `output_instructions`. Your output should follow the \
the output instructions EXACTLY.

output_instructions: {output_instructions}

user_request: ```{user_request}```

working_memory: ```{working_memory}```

Action:
"""


class Request(BaseModel):
    verb: str
    url: str
    body: Optional[str]


class Response(BaseModel):
    status_code: int
    body: Optional[str]


class Interaction(BaseModel):
    req: Request
    res: Response


class VolatileWorkingMemory(BaseModel):
    memory: List[Interaction]

    def clear(self):
        self.memory = list()

    def add(self, fragment):
        self.memory.append(fragment)

    def dump(self):
        return self.json()


class Output(BaseModel):
    action: StopAction | HAHttpRequestAction = Field(discriminator="name")


class AgentGadgets(BaseModel):
    gadgets: List[GadgetMetadata]


class ReasoningModule(ProcessingModule):
    working_memory: VolatileWorkingMemory(memory=list())
    chat: ChatOpenAI
    ha_client: HAAPIClient
    output_parser: PydanticOutputParser[Output]

    def __init__(self, ha_client: HAAPIClient, openai_api_key: str):
        self.working_memory = VolatileWorkingMemory(memory=list())
        self.chat = ChatOpenAI(model="gpt-4-1106-preview", api_key=openai_api_key)
        self.output_parser = PydanticOutputParser(pydantic_object=Output)
        self.ha_client = ha_client

    def _req(self, output: Output):
        if output.action.name == "ha_api_client":
            res = self.ha_client.req(output.action)
            self.working_memory.add(
                Interaction(
                    req=Request(
                        verb=output.action.verb,
                        url=output.action.path + output.action.query,
                        body=output.action.body,
                    ),
                    res=Response(status_code=res.status_code, body=res.content),
                )
            )

    @log
    def _parse(self, input_: str | List[str | Dict]) -> Output:
        return self.output_parser.parse(input_)

    def _invoke(self, input_: LanguageModelInput) -> BaseMessage:
        logging.info("invoking chat model... this may take a while")
        return self.chat.invoke(input_)

    def __call__(self, input_: str) -> str:
        self.working_memory.clear()
        return_text = "N/A"
        run = True

        agent_gadgets = AgentGadgets(
            gadgets=[self.ha_client.get_metadata(), StopGadget().get_metadata()]
        )

        while run:
            system_message = SystemMessage(
                content=_system_msg_template.format(
                    name="personal assistant",
                    gadgets=agent_gadgets.json(),
                    output_instructions=self.output_parser.get_format_instructions(),
                    user_request=input_,
                    working_memory=self.working_memory.dump(),
                )
            )

            resp = self._invoke([system_message])
            output: Output = self._parse(resp.content)

            if output.action.name == "ha_api_client":
                self._req(output)
            elif output.action.name == "stop":
                run = False
                return_text = output.action.reason
            else:
                RuntimeError("Uknown action name.")
        return return_text
