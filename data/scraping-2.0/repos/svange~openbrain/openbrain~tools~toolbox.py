from langchain.tools import BaseTool

from openbrain.tools.callback_handler import CallbackHandler
from openbrain.tools.protocols import OBCallbackHandlerFunctionProtocol
from openbrain.orm.model_agent_config import AgentConfig
from openbrain.orm.model_lead import Lead
from openbrain.tools.obtool import OBTool

from openbrain.util import get_logger

logger = get_logger()


class Toolbox:  # invoker
    """A toolbox for GptAgents."""

    available_tools: dict[str:OBTool] = {}

    def __init__(
        self,
        lead: Lead,
        agent_config: AgentConfig,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.callback_handler = CallbackHandler(lead=lead, agent_config=agent_config)
        # Initialize a list for the BaseTool objects and one list each for each langchain callback type
        self._initialize_known_lists()

        # Add tools listed by name in the agent_config to the list of tool, and add the tool's callbacks to the
        # appropriate callback list, creating new lists for callback types if langchain has added new callback types
        # emit a warning to the user to update the list of known callback types
        tools_to_register = agent_config.tools if agent_config.tools else [too.__name__ for too in self.available_tools.values()]

        if tools_to_register:
            for tool_name in tools_to_register:
                try:
                    obtool = self.available_tools[tool_name]
                except KeyError:
                    raise KeyError(f"Tool {tool_name} not registered\n Registered tools: {str(self.available_tools)}")

                self.callback_handler.register_ob_tool(obtool)
                self.register_obtool(obtool)

        else:
            # add all tools
            self.tools = [obtool.tool for obtool in self.available_tools.values()]

    def _initialize_known_lists(self, *args, **kwargs):
        """Initialize a list for the BaseTool objects and one list each for each langchain callback type."""
        self.tools: list[BaseTool] = []
        # Setting known langchain handler callbacks for type hinting new callbacks (i.e. on_whatever) added to
        # langchain should be added here for completeness as type hints don't work through setattr
        self.on_llm_start_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_chat_model_start_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_llm_new_token_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_llm_end_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_llm_error_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_chain_start_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_chain_end_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_chain_error_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_tool_start_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_tool_end_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_tool_error_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_text_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_agent_action_callbacks: list[OBCallbackHandlerFunctionProtocol] = []
        self.on_agent_finish_callbacks: list[OBCallbackHandlerFunctionProtocol] = []

    def register_obtool(self, obtool: type[OBTool]):
        if obtool.tool:
            self.tools.append(obtool.tool)

        # obtool_callbacks = [cb_name for cb_name in dir(obtool) if cb_name.startswith("on_")]
        #
        # for cb_name in obtool_callbacks:
        #     # create self.this_callback_list
        #     cb_list_name = f"{cb_name}_callbacks"
        #     if not getattr(self, cb_list_name):
        #         logger.WARNING(
        #             f"Callback list {cb_list_name} not found, langchain must have added new "
        #             f"callbacks to the callback handler, time to update the list of known handlers."
        #         )
        #         setattr(self, cb_list_name, [])
        #
        #     # if the callback is defined in the tool, add it to the list
        #     if getattr(obtool, cb_name):
        #         callback = getattr(obtool, cb_name)
        #         getattr(self, cb_list_name).append(callback)

    @classmethod
    def register_available_obtool(cls, tool: type[OBTool]):
        if issubclass(tool, OBTool) and tool != OBTool:
            cls.available_tools[tool.__name__] = tool
        else:
            raise TypeError(f"Tool {tool} is not a subclass of OBTool")

    def get_tools(self):
        return [tool() for tool in self.tools]
