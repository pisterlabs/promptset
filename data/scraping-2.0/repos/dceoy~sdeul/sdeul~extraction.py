#!/usr/bin/env python

import json
import logging
import os
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Optional, Union

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp, OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from .validation import read_json_schema_file

_EXTRACTION_TEMPLATE = '''\
Instructions:
- Extract only the relevant entities defined by the provided JSON schema from the input text.
- Generate the extracted entities in JSON format according to the schema.
- If a property is not present in the schema, DO NOT include it in the output.

Output format:
- Markdown code block of JSON.

Provided JSON schema:
```json
{schema}
```

Input text:
```
{input_text}
```
'''     # noqa: E501


def extract_json_from_text(
    text_file_path: str, json_schema_file_path: str,
    llama_model_file_path: Optional[str] = None,
    google_model_name: Optional[str] = 'gemini-pro',
    google_api_key: Optional[str] = None,
    openai_model_name: Optional[str] = 'gpt-3.5-turbo',
    openai_api_key: Optional[str] = None,
    openai_organization: Optional[str] = None,
    output_json_file_path: Optional[str] = None, pretty_json: bool = False,
    skip_validation: bool = False, temperature: float = 0.8,
    top_p: float = 0.95, max_tokens: int = 256, n_ctx: int = 512,
    seed: int = -1, token_wise_streaming: bool = False
) -> None:
    '''Extract JSON from input text.'''
    logger = logging.getLogger(__name__)
    if llama_model_file_path:
        llm = _read_llm_file(
            path=llama_model_file_path, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, n_ctx=n_ctx, seed=seed,
            token_wise_streaming=token_wise_streaming
        )
    else:
        overrided_env_vars = {
            'GOOGLE_API_KEY': google_api_key, 'OPENAI_API_KEY': openai_api_key,
            'OPENAI_ORGANIZATION': openai_organization
        }
        for k, v in overrided_env_vars.items():
            if v:
                logger.info(f'Override environment variable: {k}')
                os.environ[k] = v
        if google_model_name:
            llm = ChatGoogleGenerativeAI(
                model=google_model_name
            )   # type: ignore
        else:
            llm = OpenAI(model_name=openai_model_name)
    schema = read_json_schema_file(path=json_schema_file_path)
    input_text = _read_text_file(path=text_file_path)
    llm_chain = _create_llm_chain(schema=schema, llm=llm)

    logger.info('Start extracting JSON data from the input text.')
    output_string = llm_chain.invoke({'input_text': input_text})
    logger.info(f'LLM output: {output_string}')
    if not output_string:
        raise RuntimeError('LLM output is empty.')
    else:
        parsed_output_data = _parse_llm_output(
            output_string=str(output_string)
        )
        logger.info(f'Parsed output: {parsed_output_data}')
        output_json_string = json.dumps(
            obj=parsed_output_data, indent=(2 if pretty_json else None)
        )
        if skip_validation:
            logger.info('Skip validation using JSON Schema.')
        else:
            logger.info('Validate the parsed output using JSON Schema.')
            try:
                validate(instance=parsed_output_data, schema=schema)
            except ValidationError as e:
                logger.error(f'Validation failed: {output_json_string}')
                raise e
            else:
                logger.info('Validation succeeded.')
        if output_json_file_path:
            _write_file(path=output_json_file_path, data=output_json_string)
        else:
            print(output_json_string)


def _write_file(path: str, data: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f'Write data in a file: {path}')
    with open(path, 'w') as f:
        f.write(data)


def _parse_llm_output(output_string: str) -> Union[List[Any], Dict[Any, Any]]:
    logger = logging.getLogger(__name__)
    json_string = None
    markdown = True
    for r in output_string.splitlines(keepends=False):
        if json_string is None:
            if r in ('```json', '```'):
                json_string = ''
            elif r in ('[', '{'):
                markdown = False
                json_string = r + os.linesep
            else:
                pass
        elif (markdown and r != '```') or (not markdown and r):
            json_string += r + os.linesep
        else:
            break
    logger.debug(f'json_string: {json_string}')
    if not json_string:
        raise RuntimeError(f'JSON code block is not found: {output_string}')
    else:
        try:
            output_data = json.loads(json_string)
        except JSONDecodeError as e:
            logger.error(f'Failed to parse the LLM output: {output_string}')
            raise e
        else:
            logger.debug(f'output_data: {output_data}')
            return output_data


def _create_llm_chain(schema: Dict[str, Any], llm: LlamaCpp) -> LLMChain:
    logger = logging.getLogger(__name__)
    prompt = PromptTemplate(
        template=_EXTRACTION_TEMPLATE, input_variables=['input_text'],
        partial_variables={'schema': json.dumps(obj=schema)}
    )
    chain = prompt | llm | StrOutputParser()
    logger.info(f'LLM chain: {chain}')
    return chain


def _read_text_file(path: str) -> str:
    logger = logging.getLogger(__name__)
    logger.info(f'Read a text file: {path}')
    with open(path, 'r') as f:
        data = f.read()
    logger.debug(f'data: {data}')
    return data


def _read_llm_file(
    path: str, temperature: float = 0.8, top_p: float = 0.95,
    max_tokens: int = 256, n_ctx: int = 512, seed: int = -1,
    token_wise_streaming: bool = False
) -> LlamaCpp:
    logger = logging.getLogger(__name__)
    logger.info(f'Read a Llama 2 model file: {path}')
    llm = LlamaCpp(
        model_path=path, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, n_ctx=n_ctx, seed=seed,
        verbose=(
            token_wise_streaming or logging.getLogger().level <= logging.DEBUG
        ),
        callback_manager=(
            CallbackManager([StreamingStdOutCallbackHandler()])
            if token_wise_streaming else None
        )
    )
    logger.debug(f'llm: {llm}')
    return llm
