from langchain.chat_models import ChatOpenAI
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, tool
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

class ChakraAgent:
    def __init__(self, chakra_name, chakra_function, temperature=0):
        self.chakra_name = chakra_name
        self.chakra_function = chakra_function
        self.llm = ChatOpenAI(temperature=temperature)
        self.tools = []
        self.memory_key = "chat_history"
        self.system_message = SystemMessage(content=f"You are the {chakra_name} agent responsible for {chakra_function}.")
        self.prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=self.system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=self.memory_key)]
        )
        self.memory = ConversationBufferMemory(memory_key=self.memory_key, return_messages=True)
        self.agent = None
        self.agent_executor = None

    def add_tool(self, tool_function):
        """Adds a custom tool to the agent's toolset."""
        self.tools.append(tool_function)
    
    def create_tool(self, func, name):
        """Dynamically creates a tool from a function and adds it to the agent's toolset."""
        tool_func = tool(name=name)(func)
        self.add_tool(tool_func)

    def create_agent(self):
        """Creates the agent with the defined tools and prompt."""
        self.agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=self.prompt)

    def create_agent_executor(self):
        """Creates the agent executor with the agent, tools, and memory."""
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, memory=self.memory, verbose=True)

    def analyze_system_state(self, state):
        """Analyze the system state and respond accordingly. This method should be overridden in derived classes."""
        raise NotImplementedError("This method should be overridden in derived classes")

