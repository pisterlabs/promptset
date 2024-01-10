from typing import Dict, List, Optional, Tuple, Union
from langchain.agents import AgentExecutor
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.tools.base import BaseTool


class Sequence_AgentExecutor(AgentExecutor):

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return self.agent.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if self.return_intermediate_steps:
            return self.agent.return_values + ["intermediate_steps"]
        else:
            return self.agent.return_values

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:

        intermediate_steps = self._prepare_intermediate_steps(
            intermediate_steps)

        output = self.agent.plan(
            intermediate_steps,
            callbacks=run_manager.get_child() if run_manager else None,
            **inputs,
        )

        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                if result is not None and len(result) > 0:
                    a, o = result[-1]
                    observation = tool.run(
                        o,
                        verbose=self.verbose,
                        color=color,
                        callbacks=run_manager.get_child()
                        if run_manager else None,
                        **tool_run_kwargs,
                    )
                else:
                    observation = tool.run(
                        agent_action.tool_input,
                        verbose=self.verbose,
                        color=color,
                        callbacks=run_manager.get_child()
                        if run_manager else None,
                        **tool_run_kwargs,
                    )
                if tool.return_direct is False:
                    result.append((agent_action, observation))
                else:
                    return AgentFinish({"output": observation},
                                       log=observation)
        return result
