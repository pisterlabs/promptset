from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from typing import List

class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")

actor_query = "Generate the filmography for a random actor."

parser = PydanticOutputParser(pydantic_object=Actor)

# assuming we got the misformmated response i.e single quote instead of double quote
misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"
try:
    parser.parse(misformatted)
    """
    Traceback (most recent call last):
    File "/Users/seungjoonlee/git/learn-langchain/venv/lib/python3.10/site-packages/langchain/output_parsers/pydantic.py", line 27, in parse
        json_object = json.loads(json_str, strict=False)
    File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/json/__init__.py", line 359, in loads
        return cls(**kw).decode(s)
    File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/json/decoder.py", line 337, in decode
        obj, end = self.raw_decode(s, idx=_w(s, 0).end())
    File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/json/decoder.py", line 353, in raw_decode
        obj, end = self.scan_once(s, idx)
    json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
    File "/Users/seungjoonlee/git/learn-langchain/output_parsers/auto_fixing_parser.py", line 15, in <module>
        print(parser.parse(misformatted))
    File "/Users/seungjoonlee/git/learn-langchain/venv/lib/python3.10/site-packages/langchain/output_parsers/pydantic.py", line 33, in parse
        raise OutputParserException(msg, llm_output=text)
    langchain.schema.output_parser.OutputParserException: Failed to parse Actor from completion {'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}. Got: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
    """
except Exception as e:
    print("Got an error!!!!! trying with OutputFixingParser...")
    from langchain.output_parsers import OutputFixingParser

    new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
    print(new_parser.parse(misformatted))
    """
    name='Tom Hanks' film_names=['Forrest Gump']
    """
