from typing import Dict, List
from langchain.chains.base import Chain


class CaseChain(Chain):
  categorization_input: str
  subchains: Dict[str, Chain]
  default_chain: Chain
  output_variables: List[str] = ["text"]
  input_keys = []
  output_keys = []

  @property
  def input_keys(self) -> List[str]:
    keys = list(set(self.default_chain.input_keys \
                    + [key for subchain in self.subchains.values() for key in subchain.input_keys] \
                    + [self.categorization_input]))
    keys.sort()
    return keys

  @property
  def output_keys(self) -> List[str]:
    keys_set = set(self.default_chain.output_keys)
    for subchain in self.subchains.values():
      keys_set = keys_set.intersection(subchain.output_keys)
    keys = list(keys_set)
    keys.sort()
    return keys

  def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
    categorization = inputs[self.categorization_input].strip()

    subchain = self.subchains[categorization] if categorization in self.subchains else self.default_chain

    known_values = inputs.copy()
    outputs = subchain(known_values, return_only_outputs=True)
    known_values.update(outputs)
    return {k: known_values[k] for k in self.output_keys}
