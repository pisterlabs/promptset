import openai
from typing import Optional
from pydantic import BaseModel
import json

# class ContractType(Enum):
#     Erc20 = 'erc20'
#     TimeLock = 'time-lock'
#     Unsupported = 'unsupported'

supported_contracts = ["erc20", "time-lock", "custom"]


class ContractTypeAIResponse(BaseModel):
    contractType: Optional[str]


contract_type_schema = ContractTypeAIResponse.model_json_schema()
contract_type_null_object = ContractTypeAIResponse(contractType=None).model_dump_json()


def validate_contract_type_input(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=[
            {
                "name": "get_answer_for_user_query",
                "description": "Ask user what contract type is needed, unknown parameters should be null.",
                "parameters": ContractTypeAIResponse.model_json_schema(),
            }
        ],
        function_call={"name": "get_answer_for_user_query"},
    )
    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
    contract_type = output.get("contractType", None)
    if contract_type is not None and contract_type not in supported_contracts:
        return False, {
            "role": "assistant",
            "content": f"Unfortunately {contract_type} are not currently supported. The only supported contracts at the moment are {', '.join(supported_contracts)}",
        }
    return True, output
