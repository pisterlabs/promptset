import json
import os
import boto3
import logging
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain import PromptTemplate


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    body = json.loads(event.get("body", "{}"))
    logger.info(body)
    # get text from body
    input = body.get("text", "")
    system_prompts = body.get("system_prompts", [])
    # concatenate system prompts
    if isinstance(system_prompts, list):
        system_prompts = "\n".join(system_prompts)
    elif isinstance(system_prompts, str):
        system_prompts = system_prompts
    else:
        system_prompts = ""
    temperature = body.get("temperature", 0)
    logger.info("Input: " + input)
    logger.info("System Prompts: " + system_prompts)
    logger.info("temperature: " + str(temperature))
    # create prompt template
    template = (
        "System Prompts Provided are: {system_prompts}.Input provided is: {input}."
    )
    prompt = PromptTemplate(
        input_variables=["input", "system_prompts"],
        template=template,
    )
    llm = OpenAI(temperature=temperature)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    answer = llm_chain.predict(
        input=input,
        system_prompts=system_prompts,
    )
    logger.info("Answer: " + answer)
    result = {
        "statusCode": 200,
        "body": json.dumps(
            {
                "answer": answer,
            }
        ),
    }
    return result
