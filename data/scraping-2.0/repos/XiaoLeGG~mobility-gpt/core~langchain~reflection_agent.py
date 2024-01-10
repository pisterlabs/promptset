from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from core.langchain.tool_manager import collect_tools
from langchain.memory import ConversationBufferMemory
class ReflectionAgent():

    def __init__(self):

        self._llm = ChatOpenAI(temperature=0, model="gpt-4")
        self._tools = collect_tools()


        self._prompt = """Here are some information you need to know before our work:
        [Role] You are a reflection agent who is responsible for pointing out the error on the work of the mobility agent. Mobility agent is a spatio-temporal data analyst who should execute the most appropriate tools to process the data based on the request. You are going to check the executing details of mobility agent and point out the error.
        [HANDLING STEPS]
        1. Analyse the main idea and the data features mentioned in the request for the mobility agent.
        2. You are going to check the whole executing details and the log of the mobility agent.
        3. Give a feedback to the mobility agent about the error you found. If there are no errors, just return a syntax "NO_ERROR".
        [NOTICE]
        1. You are not allowed to executing any tools. The tools are only for you to understand the invoking details.
        2. The content in [REQUEST] is the user's request and the content in [MOBILITY AGENT LOG] is the details you need to check.
        """

        self._agent_memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

        # self._agent = initialize_agent(
        #     tools=self._tools,
        #     llm=self._llm,
        #     agent=AgentType.,
        #     verbose=True
        # )

        self._is_started = False
    
    def start(self):
        """Start the agent.
        
        Parameters
        ----------
        input_file : str
            The initial data file path to be processed.
        
        """
        self._is_started = True
        response = self._agent.run(self._prompt)
        return response

    def ask(self, request: str):
        """Ask the agent to solve the problem.

        Parameters
        ----------
        request : str
            The request of the user.

        Returns
        -------
        str
            The output file path.
        """
        if not self._is_started:
            raise Exception("Please start the agent first.")
        response = self._agent.run(request)
        return response