"""Chain that chooses and performs the next action."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.input import get_color_mapping
from langchain.tools.base import BaseTool

from langchain_contrib.utils import safe_inputs

from .tool import ToolChain


class ChoiceChain(Chain):
    """Chain that asks the LLM for a decision and executes it."""

    choice_picker: Chain
    """The chain that actually prompts the LLM for the choice."""
    prep_picker_output: Callable[[Dict[str, str]], Dict[str, str]] = lambda x: x
    """Interprets output from the picker chain for the chosen chain.

    Override this to do additional dict munging before it gets passed through to
    the chosen chain.
    """
    choices: Mapping[str, Chain]
    """The chains that will be run depending on the LLM's choice.

    This is a mapping from which LLM output corresponds to which chain.
    """
    choice_key: str = "choice"
    """choice_picker output key that tells us which choice was picked."""
    ignore_keys: List[str] = []
    """Keys that will be returned in final output, but not passed on to chosen chain."""
    emit_io_info: bool = False
    """If true, also returns input and output dicts for the chosen chain.

    Outputs will be returned in JSON form to preserve string function signatures.
    """
    chain_inputs_key: str = "choice_inputs"
    """Key for chosen chain inputs."""
    chain_outputs_key: str = "choice_outputs"
    """Key for chosen chain outputs."""

    @classmethod
    def from_tools(
        cls,
        choice_picker: Chain,
        tools: List[BaseTool],
        excluded_colors: List[str] = ["green"],
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChoiceChain:
        """Construct a ChoiceChain from tools.

        This also assigns colors to each tool.
        """
        from langchain_contrib.tools import ZBaseTool

        color_mapping = get_color_mapping(
            items=[str(i) for i in range(len(tools))],
            excluded_colors=excluded_colors,
        )
        wrapped_tools = [
            ZBaseTool.from_tool(
                base_tool=tool, color=color_mapping[str(i)], verbose=verbose
            )
            for i, tool in enumerate(tools)
        ]
        choices = {
            tool.name: ToolChain(tool=tool, verbose=False) for tool in wrapped_tools
        }
        return cls(
            choice_picker=choice_picker, choices=choices, verbose=verbose, **kwargs
        )

    @property
    def input_keys(self) -> List[str]:
        """Input keys to this chain."""
        return self.choice_picker.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Possible output keys produced by this chain."""
        all_keys = {self.chain_inputs_key, self.chain_outputs_key}
        all_keys.update(self.choice_picker.output_keys)
        for choice in self.choices.values():
            all_keys.update(choice.output_keys)
        return list(all_keys)

    def _validate_outputs(self, outputs: Dict[str, str]) -> None:
        """Skip validation because different options may produce different outputs.

        It is assumed that each individual choice chain will validate its own output.
        """

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        # Get picker output
        raw_picker_output = self.choice_picker(inputs, return_only_outputs=True)
        if self.choice_key not in raw_picker_output:
            raise KeyError(f"Choice-picking chain did not emit '{self.choice_key}'")
        ignored_output = {
            k: v for k, v in raw_picker_output.items() if k in self.ignore_keys
        }
        real_raw_output = {
            k: v for k, v in raw_picker_output.items() if k not in self.ignore_keys
        }
        picker_output = self.prep_picker_output(real_raw_output)
        if self.choice_key not in picker_output:
            raise KeyError(f"Picker output prepper did not emit '{self.choice_key}'")

        # Figure out choice
        choice = picker_output.pop(self.choice_key)
        assert isinstance(choice, str), f"Choice '{choice}' is not a str"
        if choice not in self.choices:
            raise KeyError(f"Choice picked does not exist: '{choice}'")
        chosen_chain = self.choices[choice]

        # Massage picker output into chosen chain inputs
        unused_args = picker_output.keys() - chosen_chain.input_keys
        if unused_args:
            raise KeyError(f"Extra input keys for choice: {unused_args}")
        full_inputs = {**inputs, **picker_output}
        chain_inputs = safe_inputs(chosen_chain, full_inputs)

        # Return chosen chain outputs
        chain_outputs = chosen_chain(chain_inputs, return_only_outputs=True)
        result = {
            self.choice_key: choice,
            **ignored_output,
            **picker_output,
            **chain_outputs,
        }
        if self.emit_io_info:
            result[self.chain_inputs_key] = json.dumps(chain_inputs)
            result[self.chain_outputs_key] = json.dumps(chain_outputs)
        return result

    def chosen_inputs(self, outputs: Dict[str, str]) -> Dict[str, str]:
        """Extract the inputs to the chosen chain from ChoiceChain output."""
        return json.loads(outputs[self.chain_inputs_key])

    def chosen_outputs(self, outputs: Dict[str, str]) -> Dict[str, str]:
        """Extract the outputs from the chosen chain from ChoiceChain output."""
        return json.loads(outputs[self.chain_outputs_key])
