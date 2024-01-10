from typing import Dict, List
from langchain.chains.base import Chain


class RecordingChain(Chain):
  recorded_calls: List[tuple[Dict[str, str], Dict[str, str]]] = []
  chain_spec_id: int
  chain: Chain

  def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
    output = self.chain._call(inputs)
    self.recorded_calls.append((inputs, output))
    return output

  @property
  def input_keys(self) -> List[str]:
    return self.chain.input_keys

  @property
  def output_keys(self) -> List[str]:
    return self.chain.output_keys

  @property
  def calls(self) -> List[tuple[Dict[str, str], Dict[str, str]]]:
    return self.recorded_calls

  def reset(self):
    self.recorded_calls.clear()
