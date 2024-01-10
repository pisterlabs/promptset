from __future__ import annotations

import datetime
from ast import literal_eval
from typing import Any

import boto3
from botocore.exceptions import ParamValidationError
from langchain.tools.base import BaseTool
from pydantic import BaseModel, Extra, Field

from openbrain.orm.model_agent_config import AgentConfig
from openbrain.tools.obtool import OBTool
from openbrain.agents.exceptions import AgentToolIncompleteLeadError
from openbrain.orm.event_lead import LeadEvent
from openbrain.orm.model_lead import Lead
from openbrain.util import config, get_logger, Defaults
from openbrain.tools.protocols import OBCallbackHandlerFunctionProtocol

logger = get_logger()

TOOL_NAME = "connect_with_agent"


# Utility classes and functions
class LeadAdaptor(BaseModel):
    class Config:
        extra = Extra.allow
        populate_by_name = True
        # validate_assignment = True

    # first_name: Optional[str] = Field(default=None)
    # middle_name: Optional[str] = Field(default=None)
    # last_name: Optional[str] = Field(default=None)
    full_name: str | None = Field(default=None)
    current_medications: list[str] | None = Field(default=None)
    date_of_birth: str | None = Field(default=None)
    email_address: str | None = Field(default=None)
    phone_number: str | None = Field(default=None)
    state_of_residence: str | None = Field(default=None)


def send_event(lead_event: LeadEvent, *args, **kwargs) -> Any:
    """Send lead to Lead Momentum."""
    logger.debug(f"Sending lead to Lead Momentum: {lead_event.__dict__}")

    # Send event to eventbus
    event_bus_friendly_name = config.EVENTBUS_NAME
    event_bus_client = boto3.client("events")
    response = None
    entries = [
        {
            "EventBusName": event_bus_friendly_name,
            "Source": Defaults.OB_TOOL_EVENT_SOURCE.value,
            "DetailType": Defaults.OB_TOOL_EVENT_DETAIL_TYPE.value,
            "Detail": lead_event.to_json(),
            "Time": datetime.datetime.now().isoformat(),
        }
    ]
    try:
        response = event_bus_client.put_events(Entries=entries)
        print(response["Entries"])
    except ParamValidationError:
        if config.OB_MODE == Defaults.OB_MODE_LOCAL.value:
            print(f"LOCAL_MODE: Can't sendto CRM in local MODE, skipping.")
        else:
            raise

    return response


# LangChain tool
class ConnectWithAgentTool(BaseTool):
    name = TOOL_NAME
    description = """Useful when you want to get the user in touch with a human agent."""
    args_schema: type[BaseModel] = LeadAdaptor
    handle_tool_error = True
    verbose = True

    def _run(self, *args, **kwargs) -> str:
        # This seemingly does nothing. All the work is done in the callback handler. This function is here for
        # the metadata.
        response = (
            "Successfully connected with agent! continue to be a helpful, and friendly assistant focused on "
            "answering health insurance questions!"
        )
        return response

    def _arun(self, ticker: str):
        raise NotImplementedError("connect_with_agent does not support async")


# on_tool_start
def on_tool_start(lead: Lead, agent_config: AgentConfig, input_str: str, *args, **kwargs) -> Any:
    """Function to run during callback handler's on_llm_start event."""

    lead_info_from_conversation = literal_eval(input_str)
    for key, new_value in lead_info_from_conversation.items():
        try:
            old_value = getattr(lead, key)
        except NameError:
            old_value = None

        if old_value == new_value:
            logger.warning(
                f"Ambiguous property for lead, arbitrarily replacing with new value.\n{old_value=}\n{new_value=}"  # TODO good place to test accuracy of the LLM agent
            )

        setattr(lead, key, new_value)

    if not lead.email_address and not lead.phone_number:
        raise AgentToolIncompleteLeadError(
            "No email or phone number provided, ask the user for an email address or phone " "number and try again."
        )
    lead.sent_by_agent = True
    lead.save()

    lead_event = LeadEvent(lead=lead, agent_config=agent_config)

    # send_business_sns(f"New lead sent: {lead_from_db=}")
    send_event(lead_event)

    return lead


def on_tool_error(lead: Lead = None, agent_config: AgentConfig = None, agent_input=None, *args, **kwargs) -> Any:
    lead.refresh()
    lead.status = "ERROR"
    lead.save()


class OBToolSendToCRM(OBTool):
    name: str = TOOL_NAME
    tool: BaseTool = ConnectWithAgentTool
    on_tool_start: OBCallbackHandlerFunctionProtocol = on_tool_start
    on_tool_error: OBCallbackHandlerFunctionProtocol = on_tool_error
