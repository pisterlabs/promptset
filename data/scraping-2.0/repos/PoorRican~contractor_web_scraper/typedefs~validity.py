from enum import Enum
from langchain.output_parsers import EnumOutputParser


class Validity(Enum):
    """ A type alias for a validity state """
    valid = 'valid'
    invalid = 'invalid'


ValidityParser = EnumOutputParser(enum=Validity)
