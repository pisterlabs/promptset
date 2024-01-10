from typing import Dict

from langchain.chains.sequential import SequentialChain
from pydantic import root_validator


class LaxSequentialChain(SequentialChain):
    @root_validator(pre=True)
    def validate_chains(cls, values: Dict) -> Dict:
        return values

    def _validate_outputs(self, outputs: Dict[str, str]) -> None:
        pass

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        known_values = inputs.copy()
        for i, chain in enumerate(self.chains):
            outputs = chain(known_values, return_only_outputs=True)
            known_values.update(outputs)
        return known_values
