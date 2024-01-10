from langchain.callbacks.manager import Callbacks
from typing import Tuple, Any
import re
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.tools.base import BaseTool
from typing import Optional, Type, Any
import json
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAIChat

from store_tools import ProdSearchTool, CustServiceTool, DefaultTool

DEFAULT_OPENAI_API_KEY = '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'

# Set up a prompt template

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


class MyLLMSingleActionAgent(LLMSingleActionAgent):
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        if len(intermediate_steps) > 0:
            obs = json.loads(intermediate_steps[-1][1])
            if obs['type'] == 'prod_list':
                return AgentFinish(
                    return_values={"output": f'I found the following products:\n{json.dumps(obs["products"])}\n'}, log=""
                )
            elif obs['type'] == 'final_msg':
                return AgentFinish(return_values={"output": obs['msg']}, log="")
            elif obs['type'] == 'kb_src':
                intermediate_steps[-1] = [intermediate_steps[-1]
                                          [0], obs['context']]
                output = self.llm_chain.run(
                    intermediate_steps=intermediate_steps,
                    stop=self.stop,
                    callbacks=callbacks,
                    **kwargs,
                )
                if "Final Answer:" in output:
                    output = output + f'\nsources:\n{obs["src"]}'
                    return self.output_parser.parse(output)
                else:
                    return AgentFinish(return_values=
                                       {"output": f'I do not have a good answer. But this may be good reference: {obs["src"]}'},
                                       log="")

        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)


class StoreChatBot:
    # Set up the base template
    template = """you are a chatbot of a Web store to answer customer questions. You have access to the following tools:

{tools}

use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

(If there is no question or no request in the user input or it is only a greeting,
...skip Thought/Action/Action Input/Observation, and give a Final Answer to greet the user and ask how you can help them.)

Begin!

Question: {input}
{agent_scratchpad}"""

    template2 = """
    
    
    """

    def __init__(self, prod_embedding_store, faq_embedding_store, openai_api_key, verbose=False):
        # Get your embeddings engine ready
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        cst = CustServiceTool(faq_embedding_store, embeddings)
        #cst.init(faq_embedding_store, embeddings)
        pst = ProdSearchTool(prod_embedding_store, embeddings)
        tools = [pst, cst, DefaultTool()]
        tool_names = [tool.name for tool in tools]
        prompt = CustomPromptTemplate(
            template=self.template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )
        print(prompt)
        # llm = OpenAIChat(model_name='gpt-3.5-turbo', temperature=0)
        llm = OpenAI(model_name='text-davinci-003', temperature=0)
        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        output_parser = CustomOutputParser()
        agent = MyLLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=verbose)

    def answer(self, user_msg) -> str:
        return self.agent_executor(user_msg)

