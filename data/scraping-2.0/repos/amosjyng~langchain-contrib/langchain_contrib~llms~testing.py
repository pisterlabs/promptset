"""Fake LLMs for testing purposes."""
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import BaseModel


class FakeLLM(LLM, BaseModel):
    """Fake LLM wrapper for testing purposes."""

    mapped_responses: Optional[Mapping] = None
    """Mapping of prompt to predetermined responses."""
    sequenced_responses: Optional[List[str]] = None
    """List of responses to return in order.

    Useful if the prompt is too complicated to reconstruct for testing.
    """
    num_calls: int = 0
    """Keeps track of which sequenced response to return."""
    check_stops: bool = False
    """Set to true to check that stops are being set, and being set properly.
    `queries` must be modified to have the stop present.
    """

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _check_stops(self, result: str, stop: Optional[List[str]]) -> str:
        if not self.check_stops:
            return result

        assert stop is not None, "Stop has not been set"
        found_stop = False
        for s in stop:
            if result.endswith(s):
                found_stop = True
                result = result[: len(result) - len(s)]
                break
        assert found_stop, f"Output '{result}' does not end in {stop}"
        return result

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        if self.sequenced_responses is not None and self.num_calls < len(
            self.sequenced_responses
        ):
            result = self.sequenced_responses[self.num_calls]
        elif self.mapped_responses is not None and prompt in self.mapped_responses:
            result = self.mapped_responses[prompt]
        else:
            result = self._uncached_queries(prompt, stop)

        self.num_calls += 1
        return self._check_stops(result, stop)

    def _uncached_queries(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Logic for queries that aren't hardcoded ahead of time."""
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}
