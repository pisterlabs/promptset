"""Module implements an agent that uses OpenAI's APIs function enabled API."""
from typing import Any, List, Optional, Sequence, Tuple, Union

from langchain.agents import BaseSingleActionAgent
# from langchain.agents.format_scratchpad.openai_functions import (
#     format_to_openai_function_messages,
# )
# from langchain.agents.output_parsers.openai_functions import (
#     OpenAIFunctionsAgentOutputParser,
# )
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
# from langchain.prompts.chat import (
#     BaseMessagePromptTemplate,
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     MessagesPlaceholder,
# )
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import root_validator
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BasePromptTemplate,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
# from langchain.tools.render import format_tool_to_openai_function

from Console.client.chains.parsers.json import JsonOutputParser
# from .chat_models.mistral_ia import ChatMistralAI

from Console.client.chains.prompts.airoboros import JSON_FUNC

def format_intermediate_steps_to_agent_scratchpad(intermediate_steps):
    """
    Formats the intermediate steps into the agent scratchpad format.

    Args:
        intermediate_steps (list): A list of tuples containing the action and observation for each intermediate step.

    Returns:
        str: The formatted agent scratchpad.
    """
    agent_scratchpad = []
    
    for action, observation in intermediate_steps:
        agent_scratchpad.extend([
            "<|im_start|>assistant",
            "{{ 'action': '{action.tool}',  'action_input': '{action.tool_input}'}}<|im_stop|>",
            "<|im_start|>observation",
            "{observation}<|im_stop|>"
        ])
    
    return "\n".join(agent_scratchpad)

class DolphinMistralAIFunctionsAgent(BaseSingleActionAgent):
    """An Agent driven by OpenAIs function powered API.

    Args:
        llm: Preferably Dolphin MistralIA 7B.
        tools: The tools this agent has access to.
        prompt: The prompt for this agent, should support agent_scratchpad as one
            of the variables. For an easy way to construct this prompt, use
            `MistralAIFunctionsAgent.create_prompt(...)`
    """

    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    prompt: BasePromptTemplate

    def get_allowed_tools(self) -> List[str]:
        """Get allowed tools."""
        return [t.name for t in self.tools]

    # @root_validator
    # def validate_llm(cls, values: dict) -> dict:
    #     if not isinstance(values["llm"], ChatMistralAI):
    #         raise ValueError("Only supported with ChatMistralAI models.")
    #     return values

    @root_validator
    def validate_prompt(cls, values: dict) -> dict:
        prompt: BasePromptTemplate = values["prompt"]
        if "agent_scratchpad" not in prompt.input_variables:
            raise ValueError(
                "`agent_scratchpad` should be one of the variables in the prompt, "
                f"got {prompt.input_variables}"
            )
        return values

    @property
    def input_keys(self) -> List[str]:
        """Get input keys. Input refers to user input here."""
        return ["input"]

    # @property
    # def functions(self) -> List[dict]:
    #     return [dict(self.format_tool(t)) for t in self.tools]

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date, along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        agent_scratchpad = format_intermediate_steps_to_agent_scratchpad(intermediate_steps)
        selected_inputs = {
            k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"
        }
        full_inputs = {"agent_scratchpad": agent_scratchpad, **selected_inputs}
        prompt = self.prompt.format_prompt(**full_inputs)
        # messages = prompt.to_messages()
        prompt_context = prompt.to_string()
        predicted_action = self.llm.predict(
                prompt_context,
                callbacks=callbacks,
                stop=["<|im_stop|>", "<|im_start|>"],
            )
        output_parser = JsonOutputParser()
        agent_decision = output_parser.parse(
            text=predicted_action
        )
        return agent_decision

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        agent_scratchpad = format_intermediate_steps_to_agent_scratchpad(intermediate_steps)
        selected_inputs = {
            k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"
        }
        full_inputs = {"agent_scratchpad": agent_scratchpad, **selected_inputs}
        prompt = self.prompt.format_prompt(**full_inputs)
        # messages = prompt.to_messages()
        prompt_context = prompt.to_string()
        predicted_action = await self.llm.apredict(
            prompt_context, callbacks=callbacks, stop=["</s>", "[INST]", "[/INST]", "<s>", "```\n"],
        )
        output_parser = JsonOutputParser()
        agent_decision = output_parser.parse(
            text=predicted_action
        )
        return agent_decision

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."}, ""
            )
        elif early_stopping_method == "generate":
            # Generate does one final forward pass
            agent_decision = self.plan(
                intermediate_steps, **kwargs
            )
            if not isinstance(agent_decision, AgentFinish):
                raise ValueError(
                    f"got AgentAction with no functions provided: {agent_decision}"
                )
            return agent_decision
        else:
            raise ValueError(
                "early_stopping_method should be one of `force` or `generate`, "
                f"got {early_stopping_method}"
            )

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system: Optional[str] = "A chat session between the User and their Assistant.",
        extra_prompt: Optional[List[str]] = None,
    ) -> BasePromptTemplate:
        """Create prompt for this agent.

        Args:
            system: Message to use as the system that will be the
                first in the prompt.
            extra_prompt: Prompt that will be placed between the
                system and the new human input.

        Returns:
            A prompt template to pass into this agent.
        """
        _prompts = extra_prompt or []
        messages: List[str]
        if system:
            messages = ["<|im_start|>system", f"{system}<|im_end|>"]
        else:
            messages = []

        messages.extend(
            [
                # History, context, whatever...
                *_prompts,
                
                # Last User Input
                "<|im_start|>user", f"{JSON_FUNC.format(query='{input}', functions=cls.get_func_str(tools))}<|im_end|>",
                
                # What has the agent acomplished yet
                "{agent_scratchpad}",
                
                # This should always be at the end of the prompt
                "<|im_start|>assistant",
                "",
            ]
        )
        return PromptTemplate(template="\n".join(messages), input_variables=["input"])

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        extra_prompt: Optional[List[str]] = None,
        system: Optional[str] = "A chat session between the User and their Assistant.",
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        """Construct an agent from an LLM and tools."""
        prompt = cls.create_prompt(
            extra_prompt=extra_prompt,
            system=system,
            tools=tools,
        )
        return cls(
            llm=llm,
            prompt=prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )
        
    @staticmethod
    def get_func_str(tools):
        return "\n".join([f"{tool.name}\n   {tool.description}" for tool in tools])

