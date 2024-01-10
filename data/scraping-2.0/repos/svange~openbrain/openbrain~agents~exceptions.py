from langchain.tools.base import ToolException

from openbrain.util import config, get_logger, get_metrics, get_tracer

logger = get_logger()
metrics = get_metrics()
tracer = get_tracer()


class AgentError(Exception):
    """Raised when the agent fails."""


class AgentToolError(ToolException):
    """Raised when the backend fails."""


class AgentToolIncompleteLeadError(AgentToolError):
    """Raised when the agent tries to connect to an agent on behalf of a lead that is not complete"""


class AgentToolLeadMomentumError(AgentToolError):
    """Raised when LeadMomentum returns an error"""

    pass
    # event_bus_name = config.EVENTBUS_FRIENDLY_NAME
    # event_source = Util.PROJECT
    # event_bridge_client = Util.BOTO_SESSION.client("events")
    # event = {
    #     "Source": event_source,
    #     "DetailType": __name__,
    #     "EventBusName": event_bus_name,
    #     "Detail": "{}",
    # }

    # TODO: Send to dead letter queue
