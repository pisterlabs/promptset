"""Util that calls APITable."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env

from apitable_toolkit.tool.prompt import (
    APITABLE_GET_FIELD_PROMPT,
    APITABLE_CREATE_FIELD_PROMPT,
    APITABLE_GET_NODES_PROMPT,
    APITABLE_GET_RECORDS_PROMPT,
    APITABLE_GET_SPACES_PROMPT,
    APITABLE_CREATE_DATASHEET_PROMPT,
)

import json
import re
import tiktoken


enc = tiktoken.encoding_for_model("text-davinci-003")


def count_tokens(s):
    return len(enc.encode(s))


# TODO: think about error handling, more specific api specs
class APITableAPIWrapper(BaseModel):
    """Wrapper for APITable API."""

    apitable: Any  #: :meta private:
    apitable_api_token: Optional[str]
    apitable_api_base: Optional[str] = "https://apitable.com"

    operations: List[Dict] = [
        {
            "mode": "get_spaces",
            "name": "Get Spaces",
            "description": APITABLE_GET_SPACES_PROMPT,
        },
        {
            "mode": "get_nodes",
            "name": "Get Nodes",
            "description": APITABLE_GET_NODES_PROMPT,
        },
        {
            "mode": "get_fields",
            "name": "Get Fields",
            "description": APITABLE_GET_FIELD_PROMPT,
        },
        {
            "mode": "create_fields",
            "name": "Create Fields",
            "description": APITABLE_CREATE_FIELD_PROMPT,
        },
        {
            "mode": "get_records",
            "name": "Get Records",
            "description": APITABLE_GET_RECORDS_PROMPT,
        },
        {
            "mode": "create_datasheets",
            "name": "Create Datasheets",
            "description": APITABLE_CREATE_DATASHEET_PROMPT,
        },
        # {
        #     "mode": "other",
        #     "name": "Catch all APITable API call",
        #     "description": APITABLE_CATCH_ALL_PROMPT,
        # },
    ]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def list(self) -> List[Dict]:
        return self.operations

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""

        apitable_api_token = get_from_dict_or_env(
            values, "apitable_api_token", "APITABLE_API_TOKEN"
        )
        values["apitable_api_token"] = apitable_api_token

        apitable_api_base = get_from_dict_or_env(
            values, "apitable_api_base", "APITABLE_API_BASE"
        )
        values["apitable_api_base"] = apitable_api_base

        try:
            from apitable import Apitable
        except ImportError:
            raise ImportError(
                "apitable is not installed. "
                "Please install it with `pip install apitable`"
            )

        apitable = Apitable(token=apitable_api_token, api_base=apitable_api_base)
        values["apitable"] = apitable

        return values

    def trans_key(self, field_key_map, key: str):
        """
        When there is a field mapping, convert the mapped key to the actual key
        """
        if key in ["_id", "recordId"]:
            return key
        if field_key_map:
            _key = field_key_map.get(key, key)
            return _key
        return key

    def query_parse(self, field_key_map, **kwargs) -> str:
        query_list = []
        for k, v in kwargs.items():
            # Handling null
            if v is None:
                v = "BLANK()"
            # Handling string
            elif isinstance(v, str):
                v = f'"{v}"'
            elif isinstance(v, bool):
                v = "TRUE()" if v else "FALSE()"
            # Handling array type values, multiple select, members?
            elif isinstance(v, list):
                v = f'"{", ".join(v)}"'
            query_list.append(f"{{{self.trans_key(field_key_map, k)}}}={v}")
        if len(query_list) == 1:
            return query_list[0]
        else:
            qs = ",".join(query_list)
            return f"AND({qs})"

    def get_spaces(self) -> str:
        spaces = self.apitable.spaces.all()
        parsed_spaces = [json.loads(space.json()) for space in spaces]
        parsed_spaces_str = (
            "Found " + str(len(parsed_spaces)) + " spaces:\n" + str(parsed_spaces)
        )
        return parsed_spaces_str

    def get_nodes(self, params: dict) -> str:
        try:
            space_id = params["space_id"]
        except Exception as e:
            return f"Found a error 'Action input need effective space_id', please try another tool to get right space_id."
        try:
            nodes = self.apitable.space(space_id=space_id).nodes.all()
            parsed_nodes = [json.loads(node.json()) for node in nodes]
            parsed_nodes_str = (
                "Found " + str(len(parsed_nodes)) + " nodes:\n" + str(parsed_nodes)
            )
        except Exception as e:
            parsed_nodes_str = (
                f"Found a error '{e}', please try another tool to get right space_id."
            )
        return parsed_nodes_str

    def get_fields(self, params: dict) -> str:
        try:
            datasheet_id = params["datasheet_id"]
        except Exception as e:
            return f"Found a error 'Action input need effective datasheet_id', please try another tool to get right datasheet_id."
        try:
            fields = self.apitable.datasheet(datasheet_id).fields.all()
            parsed_fields = [json.loads(field.json()) for field in fields]
            parsed_fields_str = (
                "Found " + str(len(parsed_fields)) + " fields:\n" + str(parsed_fields)
            )
        except Exception as e:
            parsed_fields_str = f"Found a error '{e}', please try another tool to get right datasheet_id."
        return parsed_fields_str

    def create_fields(self, params: dict) -> str:
        try:
            space_id = params["space_id"]
            datasheet_id = params["datasheet_id"]
            field_data = params["field_data"]
        except Exception as e:
            return f"Found a error '{e}', please try correct it"

        try:
            field = (
                self.apitable.space(space_id)
                .datasheet(datasheet_id)
                .fields.create(field_data)
            )
            parsed_fields_str = "Field created! \n" + str(field.json())
        except Exception as e:
            parsed_fields_str = (
                f"Found a error '{e}', please try to correct field_data."
            )
        return parsed_fields_str

    def create_datasheets(self, params: dict) -> str:
        try:
            space_id = params["space_id"]
            name = params["name"]
            field_data = params["field_data"]
        except Exception as e:
            return f"Found a error 'Action input need effective {e}', please generate effective {e} and try again"

        try:
            field = self.apitable.space(space_id).datasheets.create(
                {"name": name, "fields": field_data}
            )
            parsed_fields_str = "Datasheet created! \n" + str(field.json())
        except Exception as e:
            parsed_fields_str = (
                f"Found a error '{e}', please try to correct field_data."
            )
        return parsed_fields_str

    def get_records(self, params: dict) -> str:
        try:
            datasheet_id = params["datasheet_id"]
        except Exception as e:
            return f"Found a error 'Action input need effective datasheet_id', please try another tool to get right datasheet_id."
        dst = self.apitable.datasheet(datasheet_id)
        kwargs = {}
        if "filter_condition" in params:
            query_formula = self.query_parse(params["filter_condition"])
            kwargs["filterByFormula"] = query_formula
        if "sort_condition" in params:
            kwargs["sort"] = params["sort_condition"]
        if "maxRecords_condition" in params:
            kwargs["maxRecords"] = params["maxRecords_condition"]
        try:
            records = dst.records.all(**kwargs)
            parsed_records = [record.json() for record in records]
            parsed_records_str = (
                "Found "
                + str(len(parsed_records))
                + " records:\n"
                + str(parsed_records)
            )
            if count_tokens(parsed_records_str) > 1000:
                parsed_records_str = (
                    "Found "
                    + str(len(parsed_records))
                    + " records, too many to show. Try to use maxRecords or filter_condition to limit"
                )
        except Exception as e:
            if str(e) == "The sorted field does not exist":
                parsed_records_str = f"Found a error '{e}', please try another tool to get right field name."
            elif str(e) == "api_param_formula_error":
                parsed_records_str = (
                    f"Found a error '{e}', please try to make right filter_condition."
                )
            else:
                parsed_records_str = f"Found a error '{e}', please try another tool."
        return parsed_records_str

    def run(self, mode: str, query: str) -> str:
        if mode == "get_spaces":
            return self.get_spaces()

        try:
            params = json.loads(query)
        except Exception as e:
            return f"Found a error '{e}', you should only output Action Input in JSON format"

        if "space_id" in params:
            pattern = r"^spc"
            if re.match(pattern, params["space_id"], re.IGNORECASE):
                pass
            else:
                return f"Found a error 'Action input need effective space_id', please try another tool to get right space_id."

        if mode == "get_nodes":
            return self.get_nodes(params)
        elif mode == "get_fields":
            return self.get_fields(params)
        elif mode == "create_fields":
            return self.create_fields(params)
        elif mode == "get_records":
            return self.get_records(params)
        elif mode == "create_datasheets":
            return self.create_datasheets(params)
        else:
            raise ValueError(f"Got unexpected mode {mode}")
