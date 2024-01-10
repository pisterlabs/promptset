from langchain.output_parsers.enum import EnumOutputParser
from enum import Enum

class Colors(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

parser = EnumOutputParser(enum=Colors)
# # Colors.RED
# parser.parse("red")

# # Can handle spaces
# parser.parse(" green")

# # And new lines
# parser.parse("blue\n")

# # And raises errors when appropriate
# parser.parse("yellow")

# """
# Traceback (most recent call last):
# File "/Users/seungjoonlee/git/learn-langchain/venv/lib/python3.10/site-packages/langchain/output_parsers/enum.py", line 27, in parse
# return self.enum(response.strip())
# File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/enum.py", line 385, in __call__
# return cls.__new__(cls, value)
# File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/enum.py", line 710, in __new__
# raise ve_exc
# ValueError: 'yellow' is not a valid Colors

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
# File "/Users/seungjoonlee/git/learn-langchain/output_parsers/enum_parser.py", line 20, in <module>
# parser.parse("yellow")
# File "/Users/seungjoonlee/git/learn-langchain/venv/lib/python3.10/site-packages/langchain/output_parsers/enum.py", line 29, in parse
# raise OutputParserException(
# langchain.schema.output_parser.OutputParserException: Response 'yellow' is not one of the expected values: ['red', 'green', 'blue']
# """