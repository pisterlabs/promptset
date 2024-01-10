from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, ValidationError, HttpUrl, validator, EmailStr
from typing import NoReturn, Callable, Annotated, TypeVar

from typedefs.address import Address
from typedefs.validity import ValidityParser, Validity
from llm import LLM


class BaseContractor(BaseModel):
    title: Annotated[str, Field(description='Title of the contractor')]
    description: Annotated[str, Field(description='Description of the contractor')]
    url: Annotated[HttpUrl, Field(description='URL of the contractor homepage')]
    # TODO: add industry field

    @classmethod
    @validator('description')
    def is_contractor(cls, v) -> str:
        prompt = PromptTemplate.from_template(
            """Does this description describe an actual construction contractor website? {description}
            
            {format_instructions}
            """,
            partial_variables={'format_instructions': ValidityParser.get_format_instructions()},
            output_parser=ValidityParser
        )
        chain = prompt | LLM | ValidityParser
        response: Validity = chain.invoke(v)
        if response.valid:
            raise ValidationError(f"Invalid URL: {v}")
        return v


class Contractor(BaseContractor):
    """ Abstraction for parsed contractor data """
    phone: str | None = Field(description='Phone number of the contractor')
    email: EmailStr | None = Field(description='Email address of the contractor')
    address: Address | None = Field(description='Physical mailing address of the contractor')

    @staticmethod
    def fields() -> list[str]:
        return list(Contractor.__fields__.keys())

    def set_address(self, address: Address) -> NoReturn:
        """ Update address object

        This is to be used as a callback from future scraper classes.
        """
        # TODO: eventually, this should accept a list of addresses
        self.address = address

    def set_phone(self, phone: str) -> NoReturn:
        """ Update phone string

        This is to be used as a callback from future scraper classes.
        """
        self.phone = phone

    def set_email(self, email: str) -> NoReturn:
        """ Update email string

        This is to be used as a callback from future scraper classes.
        """
        self.email = email

    def __repr__(self) -> str:
        return f"<Contractor: {self.title}; url: {self.url}; description: {self.description}>"

    def pretty(self):
        msg = f"Contractor: {self.title}\n"
        msg += f"URL: {self.url}\n"
        msg += f"Description: {self.description}\n"

        if self.phone:
            msg += f"Phone: {self.phone}\n"
        if self.email:
            msg += f"Email: {self.email}\n"
        if self.address:
            msg += f"Address: {self.address}\n"

        return msg


T = TypeVar('T')


ContractorCallback = Callable[[T], NoReturn]
""" A callback function to set a `Contractor` attribute. """
