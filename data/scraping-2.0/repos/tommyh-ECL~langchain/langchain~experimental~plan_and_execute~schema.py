from abc import abstractmethod
from typing import List, Tuple

from pydantic import BaseModel, Field

from langchain.schema import BaseOutputParser


class Step(BaseModel):
    value: str

    # NOTE: TFH - I'm using this to improve the formatting of the insertion of step information into the prompt
    # this should probably be done at the parser level in the future
    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


class Plan(BaseModel):
    steps: List[Step]


class StepResponse(BaseModel):
    response: str

    # NOTE: TFH - I'm using this to improve the formatting of the insertion of step information into the prompt
    # this should probably be done at the parser level in the future
    def __repr__(self):
        return self.response

    def __str__(self):
        return self.response


class BaseStepContainer(BaseModel):
    @abstractmethod
    def add_step(self, step: Step, step_response: StepResponse) -> None:
        """Add step and step response to the container."""

    @abstractmethod
    def get_final_response(self) -> str:
        """Return the final response based on steps taken."""


class ListStepContainer(BaseModel):
    steps: List[Tuple[Step, StepResponse]] = Field(default_factory=list)

    def __repr__(self):
        return self._get_output_string()

    def __str__(self):
        return self._get_output_string()

    def _get_output_string(self):
        output_string = "".join(f'Step #{num+1}\n\tInput: {step}\n\tResponse: {response}' for num, (step, response) in enumerate(self.steps))
        return output_string

    def add_step(self, step: Step, step_response: StepResponse) -> None:
        self.steps.append((step, step_response))

    def get_steps(self) -> List[Tuple[Step, StepResponse]]:
        return self.steps

    def get_final_response(self) -> str:
        return self.steps[-1][1].response


class PlanOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> Plan:
        """Parse into a plan."""
