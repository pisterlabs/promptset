import openai
import json
from typing import List, Optional
from pydantic import BaseModel
from solcx import compile_source, compile_standard, compile_files
import tempfile
import subprocess
import os


class CodeGenAIResponse(BaseModel):
    code: Optional[str]


contract_type_schema = CodeGenAIResponse.model_json_schema()
contract_type_null_object = CodeGenAIResponse(code=None).model_dump_json()


def generate_code(messages):
    if len(messages) == 0:
        return {"role": "assistant", "content": "Please describe what the functionality of the smart contract should be"}
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "Write code a solidity smart contract that implements what the user describes"},
        ] + list(filter(lambda x: x.get("role").lower() != "system", messages)),
        functions=[
            {
            "name": "get_answer_for_user_query",
            "description": "Generate the solidity code the user asked for",
            "parameters": CodeGenAIResponse.model_json_schema()
            }
        ],
        function_call={"name": "get_answer_for_user_query"}
    )

    try:
        code = json.loads(response.choices[0]["message"]["function_call"]["arguments"])["code"]
    except:
        broken_json = response.choices[0]["message"]["function_call"]["arguments"]
        pragma_start = broken_json.find("pragma")
        last_quote = broken_json.rfind('"')
        code = broken_json[pragma_start:last_quote]

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(code.encode())
    tmp.close()

    result = subprocess.check_output(['myth', 'analyze', tmp.name]).decode()

    issues = []
    if "The analysis was completed successfully. No issues were detected." in result:
        issues = []
    elif result.startswith("mythril.interfaces.cli [ERROR]"):
        issues.append("Security check failed, please rerun or manually audit the generated code")
    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "system", "content": f"""
                    Explain the following mythril analysis result.
                    {result}
                    """},
            ],
        )
        issues.append(response["choices"][0]["message"]["content"])

    compiled_sol = {}
    try:
        compiled_sol = compile_files([tmp.name], output_values=["abi", "bin"], solc_version="0.8.20", import_remappings=["@openzeppelin=./openzeppelin-contracts-5.0.0"])
    except:
        issues.append("Failed to compile")

    binary = ""
    for k, v in compiled_sol.items():
        if tmp.name in k:
            binary = v.get("bin")

    os.unlink(tmp.name)

    return {
        "src": code,
        "bin": binary,
        "issues": issues
    }
