from langchain import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.agents import TrajectoryEvalChain
from src.agent.prompt_template import (ReActOutputParser, ReActPromptTemplate,
                                       T5OutputParser, T5PromptTemplate)
from src.agent.retriever import ToolRetriever


class UI_info:
    wait_user_inputs: bool = False
    user_inputs: str = ""
    chatbots: list = []
    intermediate_step_index: int = 0
    travel_plans: list = []


class Agent:
    """The agent generates reasoning path and calls APIs"""

    def __init__(self, tool_path, agent_id=0, model_name="gpt-3.5-turbo-16k-0613",
                 debug=False, visualize_trajectory=True, mode="T5"):
        self.agent_id = agent_id
        self.debug = debug
        self.UI_info = UI_info()
        self.intermediate_step_index = 0
        self.visualize_trajectory = visualize_trajectory
        self.llm = ChatOpenAI(temperature=0, model_name=model_name, verbose=self.debug)
        self.retriever = ToolRetriever(tool_path=tool_path)
        self.tools = self.retriever.get_tools()

        self.match_mode(mode)  # Pick mode

        self.started = False
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.Prompt_template, verbose=self.debug)
        tool_names = [tool.name for tool in self.tools]
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            stop=["Tool output:"] if mode == "T5" else ["Observation:"],
            output_parser=self.output_parser,
            allowed_tools=tool_names,
            verbose=self.debug,
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=self.visualize_trajectory,
            max_iterations=100,
            max_execution_time=60 * 60
        )
        self.eval_llm = ChatOpenAI(temperature=0, model_name="gpt-4-0613")
        self.eval_chain = TrajectoryEvalChain.from_llm(
            llm=self.eval_llm,  # Note: This must be a chat model
            agent_tools=self.tools,
            return_reasoning=True,
        )

    def match_mode(self, mode):
        match mode:
            case "T5":
                with open("src/prompts/Format/T5prompt.txt") as f:
                    self.T5_prompt = f.read()
                self.Prompt_template = T5PromptTemplate(
                    template=self.T5_prompt,
                    tools=self.tools,
                    input_variables=["input", "intermediate_steps"]
                )
                self.output_parser = T5OutputParser()
            case "travel":
                with open("src/prompts/Travel/make_one_day_itinerary_prompt.txt") as f:
                    self.travel_prompt = f.read()
                self.Prompt_template = ReActPromptTemplate(
                    template=self.travel_prompt,
                    tools=self.tools,
                    input_variables=["famous_sights", "restaurants", "hotel", "intermediate_steps"]
                )
                self.output_parser = ReActOutputParser()
            case "ReAct_chat":
                with open("src/prompts/Format/ReAct_chat_prompt.txt") as f:
                    self.ReAct_chat_prompt = f.read()
                self.Prompt_template = ReActPromptTemplate(
                    template=self.ReAct_chat_prompt,
                    tools=self.tools,
                    input_variables=["input", "intermediate_steps"]
                )
                self.output_parser = ReActOutputParser()
            case _:
                with open("src/prompts/Format/ReAct_prompt.txt") as f:
                    self.ReAct_prompt = f.read()
                self.Prompt_template = ReActPromptTemplate(
                    template=self.ReAct_prompt,
                    tools=self.tools,
                    input_variables=["input", "intermediate_steps"]
                )
                self.output_parser = ReActOutputParser()

    def kill_agent(self):
        self.agent_executor.killed = True
        self.UI_info = UI_info()

    def get_intermediate_steps(self):
        return self.agent_executor.intermediate_steps

    def run(self, user_input):
        self.started = True
        return self.agent_executor(inputs=user_input, return_only_outputs=True)
