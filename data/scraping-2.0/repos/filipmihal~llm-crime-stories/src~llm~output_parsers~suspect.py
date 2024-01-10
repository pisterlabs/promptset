import json
from json.decoder import JSONDecodeError
from langchain.schema import BaseOutputParser
from marshmallow import ValidationError
import re
from typing import Optional

from llm.marshmallow.schemas.suspect import SuspectSchema


class SuspectJsonOutputParser(BaseOutputParser):
    """
    Parse the output of an LLM call of the Suspect chain to JSON.
    """

    def parse(self, text: str) -> Optional[SuspectSchema]:
        """
        Parse the output of an LLM call.
        """
        try:
            objs = re.findall(r'\{[^{}]*\}', text)
            objs = [json.loads(o) for o in objs]
            objs = [{k.strip():v for k, v in o.items()} for o in objs]
            
            return [SuspectSchema().load(o) for o in objs]
        except JSONDecodeError as decode_err:
            print(decode_err)
            return None
        except ValidationError as err:
            print(err.messages)
            return None
