import logging
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.schema import OutputParserException

logger = logging.getLogger(__name__)

template = """You're an experienced smart contract developer.

You're asked to check compile errors and fix the code of smart contract.
You would grasp problems of the code from current code and compile errors first.
You are given use case, requirements, specifications, current code, and compile errors.

You must add function or variables for Undeclared identifier.
You should ALWAYS think what to do next.

Use-case: {query}

Requirements: {requirements}

Specifications: {specifications}

Source Code: ```
{source_code}
```

Compile Errors: ```
{compile_error}
```

You need to return only Solidity code and you don't need to explain.
"""

def extract_codes(text):
    if "```" in text:
        lines = text.splitlines()

        indecies = []
        for i, line in enumerate(lines):
            if "```" in line:
                indecies.append(i)

        if len(indecies) != 2:
            raise OutputParserException(
                f"Failed to parse solidity from completion {text}. Got: code is not wrapped with {{}}"
            )

        return '\n'.join(lines[indecies[0] + 1:indecies[1]])
    else:
        return text


def fix_solidity_code(query, requirements, specifications, source_code, compile_error, solidity_file_path, temperature):
    logger.info('fix_solidity_code temperature=%f, solidity_file_path=%s, query=%s, requirements=%s, specifications=%s, compile_error=%s', temperature, solidity_file_path, query, requirements, specifications, compile_error)

    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "requirements", "specifications", "source_code", "compile_error"],
    )

    model = OpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=temperature)
    _input = prompt.format_prompt(
        query=query,
        requirements=requirements,
        specifications=specifications,
        source_code=source_code,
        compile_error=compile_error
    )

    print('prompt:\n{}'.format(_input.to_string()))

    output = model(_input.to_string())

    print('output:\n{}'.format(output))

    with open(solidity_file_path, "w") as f:
        codes = extract_codes(output)
        f.write(codes)

    logger.info('fix_solidity_code completed')
