from typing import Annotated

from langchain.pydantic_v1 import BaseModel, Field, validator, ValidationError, root_validator


class Address(BaseModel):
    street: Annotated[str, Field(description="Street or mailing address for business", min_length=1)]
    city: Annotated[str, Field(description="City", min_length=1)]
    state: Annotated[str, Field(description="State", min_length=2, max_length=2)]
    zip: Annotated[int, Field(description="Zip code")]

    @root_validator(pre=True, skip_on_failure=True)
    @classmethod
    def _parsing_fail(cls, values: dict) -> bool:
        """ check if 'no address' is in the values"""
        if 'no address' in values.values():
            raise ValidationError("No address found")
        return values

    @validator('street')
    @classmethod
    def is_street(cls, v) -> str:
        """ Validate that the street is a valid mailing address """
        if v.count(' ') < 2:
            raise ValidationError("Invalid street")
        return v

    @validator('zip')
    @classmethod
    def is_zip(cls, v) -> int:
        if len(str(v)) != 5:
            raise ValidationError("Invalid zip code")
        return v
