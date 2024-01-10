import json
from typing import Union
from urllib.parse import urljoin

from langchain.tools import APIOperation, OpenAPISpec
from pydantic import BaseModel


class OpenAPICommand(BaseModel):
    command: str
    args: dict


class OpenAPIClarification(BaseModel):
    user_question: str


OpenAPIResponse = Union[OpenAPICommand, OpenAPIClarification]


class OpenAPIResponseWrapper(BaseModel):
    """Just a wrapper class for the union to ease parsing"""

    resp: OpenAPIResponse

    @staticmethod
    def parse_str(s) -> OpenAPIResponse:
        return OpenAPIResponseWrapper.parse_obj({"resp": json.loads(s)}).resp


OpenAPIOperations = dict[str, APIOperation]


def collect_operations(spec: OpenAPISpec, spec_url: str) -> OpenAPIOperations:
    operations: dict[str, APIOperation] = {}

    def add_operation(spec, path, method):
        operation = APIOperation.from_openapi_spec(spec, path, method)  # type: ignore
        operation.base_url = urljoin(spec_url, operation.base_url)
        operations[operation.operation_id] = operation

    if spec.paths is None:  # type: ignore
        return operations

    for path, path_item in spec.paths.items():  # type: ignore
        if path_item.get is not None:
            add_operation(spec, path, "get")
        if path_item.post is not None:
            add_operation(spec, path, "post")

    return operations
