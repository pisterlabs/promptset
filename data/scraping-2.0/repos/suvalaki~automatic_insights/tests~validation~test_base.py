import pytest 
from typing import List

from pydantic import BaseModel
from langchain.chains import LLMChain

from ai.validation.base import (
    ValidationSignal, ValidatorBase, Validation, Validator, 
    ParalellValidator, SequentialValidator
)


class TestInput(BaseModel):
    field:str = "test"


class TestValidatorBase(ValidatorBase[TestInput]):

    # @property
    # def input_keys(self) -> List[str]:
    #     return []

    def validate(self, _input: TestInput, **kwargs) -> Validation[TestInput]:
        return Validation(
            signal = ValidationSignal.VALID, 
            errors = []
        )

    async def avalidate(self, _input: TestInput, **kwargs) -> Validation[TestInput]:
        return Validation(
            signal = ValidationSignal.VALID, 
            errors = []
        )


def test_TestValidatorBase():

    validator = TestValidatorBase()
    v_input = TestInput()
    validator.validate(v_input)



class TestValidator(Validator[TestInput]):

    def validate(self, _input: TestInput, **kwargs) -> Validation[TestInput]:
        return Validation(
            signal = ValidationSignal.VALID, 
            errors = []
        )

    async def avalidate(self, _input: TestInput, **kwargs) -> Validation[TestInput]:
        return Validation(
            signal = ValidationSignal.VALID, 
            errors = []
        )


def test_TestValidator():

    validator = TestValidator()
    v_input = TestInput()
    validator({"input": v_input})


from ai.validation.fixer import *


class TestValidationTreater(ValidationTreater[TestInput]):

    def _treat(self, _input_object: TestInput, _validation: Validation[TestInput], **kwargs) -> TestInput:
        return TestInput(
            field="new"
        )


def test_TestValidationTreater():

    validator = TestValidator()
    v_input = TestInput()
    fixer = TestValidationTreater(validator=validator)
    validation = validator({"input": v_input})
    fixed = fixer(validation)


validator = TestValidator()
v_input = TestInput()
fixer = TestValidationTreater(validator=validator)
validation = validator({"input": v_input})
fixed = fixer(inputs=validation)