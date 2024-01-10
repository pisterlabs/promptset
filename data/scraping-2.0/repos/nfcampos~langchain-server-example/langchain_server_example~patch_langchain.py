from typing import Any, Dict, List, Tuple, Union

from langchain.agents.agent import Agent, AgentExecutor
from langchain.input import get_color_mapping
from langchain.schema import AgentAction, AgentFinish

from .context import Context

# Look for
# -- PATCH --


def patch():
    Agent.plan = _agent_plan


# Agent.plan() is called by AgentExecutor._call()
def _agent_plan(
    self: Agent,
    intermediate_steps: List[Tuple[AgentAction, str]],
    ctx: Context,
    **kwargs: Any,
) -> Union[AgentFinish, AgentAction]:
    """Given input, decided what to do.

    Args:
        intermediate_steps: Steps the LLM has taken to date,
            along with observations
        **kwargs: User inputs.

    Returns:
        Action specifying what tool to use.
    """
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
    new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
    full_inputs = {**kwargs, **new_inputs}
    full_output = self.llm_chain.predict(**full_inputs)
    # -- PATCH --
    ctx.send_message_sync(content=full_output)
    parsed_output = self._extract_tool_and_input(full_output)
    while parsed_output is None:
        full_output = self._fix_text(full_output)
        full_inputs["agent_scratchpad"] += full_output
        output = self.llm_chain.predict(**full_inputs)
        # -- PATCH --
        ctx.send_message_sync(content=full_output)
        full_output += output
        parsed_output = self._extract_tool_and_input(full_output)
    tool, tool_input = parsed_output
    if tool == self.finish_tool_name:
        return AgentFinish({"output": tool_input}, full_output)
    return AgentAction(tool, tool_input, full_output)


class AgentExecutorWithContext(AgentExecutor):
    ctx: Context = None

    class Config:
        arbitrary_types_allowed = True

    def _call(self: AgentExecutor, inputs: Dict[str, str]) -> Dict[str, Any]:
        ctx: Context = self.ctx
        """Run text through and get agent response."""
        # Do any preparation necessary when receiving a new input.
        self.agent.prepare_for_new_call()
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool.func for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # We now enter the agent loop (until it returns something).
        while True:
            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                **inputs,
                ctx=ctx,  # -- PATCH --
            )
            # If the tool chosen is the finishing tool, then we end and return.
            if isinstance(output, AgentFinish):
                # if self.verbose:
                #     self.logger.log_agent_end(output, color="green")
                final_output = output.return_values
                if self.return_intermediate_steps:
                    final_output["intermediate_steps"] = intermediate_steps
                # -- PATCH --
                ctx.send_message_sync(content=final_output["output"])
                return final_output
            # if self.verbose:
            #     self.logger.log_agent_action(output, color="green")
            # And then we lookup the tool
            if output.tool in name_to_tool_map:
                chain = name_to_tool_map[output.tool]
                # We then call the tool on the tool input to get an observation
                observation = chain(output.tool_input)
                # -- PATCH --
                ctx.send_message_sync(content=observation)
                color = color_mapping[output.tool]
            else:
                observation = f"{output.tool} is not a valid tool, try another one."
                color = None
            # if self.verbose:
            #     self.logger.log_agent_observation(
            #         observation,
            #         color=color,
            #         observation_prefix=self.agent.observation_prefix,
            #         llm_prefix=self.agent.llm_prefix,
            #     )
            intermediate_steps.append((output, observation))
