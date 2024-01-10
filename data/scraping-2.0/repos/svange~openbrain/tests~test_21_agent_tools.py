import contextlib
import datetime
import importlib
import inspect
import pkgutil
import sys

import boto3
import pytest
from langchain.callbacks.base import BaseCallbackHandler

import openbrain.tools.tool_send_lead_to_crm
from openbrain.agents.gpt_agent import GptAgent
from openbrain.orm.event_lead import LeadEvent
from openbrain.orm.model_agent_config import AgentConfig
from openbrain.orm.model_lead import Lead
from openbrain.tools.callback_handler import CallbackHandler
from openbrain.tools.tool_send_lead_to_crm import send_event
from openbrain.tools.toolbox import Toolbox
from openbrain.tools.protocols import OBCallbackHandlerFunctionProtocol
from openbrain.util import config, Defaults


@pytest.fixture
def event_bridge_client():
    client = boto3.client("events")
    return client


@pytest.fixture
def incoming_agent_config(default_agent_config):
    outgoing_agent_config = default_agent_config
    return outgoing_agent_config


@pytest.fixture
def incoming_lead(simple_lead):
    outgoing_lead = simple_lead
    return outgoing_lead


class TestAgentTools:
    """Test the GptAgent's tools."""

    @pytest.mark.tools
    def test_callback_handler_up_to_date(self):
        """Test that the callback handler protocol is up to date with langchain. If this test fails, find the langchain handler documentation, and add any missing functions to the protocol."""
        ob_callback_handler = CallbackHandler
        ob_callback_handler_functions = [
            func for func in dir(ob_callback_handler) if func.startswith("on_")
        ]

        lc_callback_handler = BaseCallbackHandler
        lc_callback_handler_functions = [
            func for func in dir(lc_callback_handler) if func.startswith("on_")
        ]

        funcs_missing_in_openbrain_callback_handler = []
        funcs_missing_in_langchain_callback_handler = []
        for function in lc_callback_handler_functions:
            if function not in ob_callback_handler_functions:
                funcs_missing_in_openbrain_callback_handler.append(function)

        for function in ob_callback_handler_functions:
            if function not in lc_callback_handler_functions:
                funcs_missing_in_langchain_callback_handler.append(function)

        assert len(funcs_missing_in_openbrain_callback_handler) == 0
        assert len(funcs_missing_in_langchain_callback_handler) == 0

    @pytest.mark.tools
    def test_protocol_up_to_date(self):
        """Test to ensure updates to the openbrain handler are always reflected in the protocol. If this test fails, find the missing parameters from the openbrain handler's class definition,and add any  missing parameters."""
        protocol_func = OBCallbackHandlerFunctionProtocol.__call__
        handler_func = CallbackHandler.__init__
        protocol_params = list(inspect.signature(protocol_func).parameters)
        handler_params = list(inspect.signature(handler_func).parameters)

        params_missing_in_protocol = []
        params_missing_in_handler = []
        for param in handler_params:
            if param not in protocol_params:
                params_missing_in_protocol.append(param)

        for param in protocol_params:
            if param not in handler_params:
                params_missing_in_handler.append(param)

        assert len(params_missing_in_protocol) == 0
        assert len(params_missing_in_handler) == 0

    @pytest.mark.tools
    def test_manual_send_to_crm_tool(
        self, incoming_agent_config: AgentConfig, incoming_lead: Lead, event_bridge_client
    ):
        """Send an event to the lead event stream."""

        lead_event = LeadEvent(lead=incoming_lead, agent_config=incoming_agent_config)
        response = send_event(lead_event=lead_event)

        event_id = response["Entries"][0]["EventId"]
        assert event_id is not None
        assert isinstance(event_id, str)
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200

    @pytest.mark.tools
    def test_tools_are_registered(self):
        """Test that the tools are registered."""
        ob_modules = sys.modules["openbrain.tools"]
        # Dynamically import all submodules

        package = sys.modules[ob_modules.__package__]
        submodules = {
            name: importlib.import_module(ob_modules.__package__ + "." + name)
            for loader, name, is_pkg in pkgutil.walk_packages(package.__path__)
        }

        # submodules = import_submodules(ob_modules.__package__)
        for submodule in submodules.values():
            for obj in inspect.getmembers(submodule):  # inspect.getmembers(submodule))
                # Broken down for troubleshooting
                with contextlib.suppress(KeyError, TypeError):
                    identified_class = obj[1]
                    Toolbox.register_available_obtool(identified_class)

        assert len(Toolbox.available_tools) > 0
        for tool_name, tool in Toolbox.available_tools.items():
            assert tool_name is not None
            assert tool is not None

    @pytest.mark.tools
    @pytest.mark.expected_failure
    @pytest.mark.skip
    def test_agent_send_to_crm_tool(
        self,
        incoming_agent_config: AgentConfig,
        incoming_lead: Lead,
        message="My name is Chad Geepeeti, my DOB is April 1, 1975. I live in Mississippi, and I currently take vicodin and tylonol. My phone number is 619-465-7894 and my email is e@my.ass. Please get me in touch with an agent immediately.",
    ):
        """Send an event to the lead event stream by getting the agent to invoke the function."""
        openbrain.tools.tool_send_lead_to_crm.fake_event_bus = {}
        results_container = {}

        def simple_mock(
            lead_event: LeadEvent, results_container: dict = results_container, *args, **kwargs
        ):
            event_bus_friendly_name = config.EVENTBUS_NAME
            event = [
                {
                    "EventBusName": event_bus_friendly_name,
                    "Source": Defaults.OB_TOOL_EVENT_SOURCE.value,
                    "DetailType": Defaults.OB_TOOL_EVENT_DETAIL_TYPE.value,
                    "Detail": lead_event.to_json(),
                    "Time": datetime.datetime.now().isoformat(),
                }
            ]

            results_container["lead_event"] = lead_event
            results_container["event"] = event
            results_container["kwargs"] = kwargs
            return lead_event

        openbrain.tools.tool_send_lead_to_crm.send_event = (
            simple_mock  # First mock ever... pretty fuckin powerful...
        )
        agent = GptAgent(agent_config=incoming_agent_config, lead=incoming_lead)
        # agent.tools[0]._run = simple_mock

        response = agent.handle_user_message(message)

        assert results_container is not None
        for key, value in results_container.items():
            assert key is not None
            assert value is not None

        assert response is not None
        # TODO assert that response is a success messge

        lead_event = results_container["lead_event"]
        assert lead_event is not None
        assert isinstance(lead_event, LeadEvent)
        assert lead_event.lead is not None
        assert lead_event.agent_config is not None

        lead_from_lead_event = lead_event.lead
        for key, val in incoming_lead.to_dict().items():
            assert getattr(lead_from_lead_event, key) == val
