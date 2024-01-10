import json
import logging
import os
import secrets
from uuid import uuid4

import boto3
import requests
from pydex_lib.authorizer import create_policy, find_resources
from pydex_lib.codex import OpenAICodex
from pydex_lib.repository import Repository
from pydex_lib.request_helper import build_pydex_error_response, build_pydex_response
from pydex_lib.request_models import Request
from pydex_lib.telegram import telegram_on_error
from pydex_lib.rate_limiter import rate_limited

stage = os.environ["STAGE"]
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

db = Repository(boto3.client("dynamodb"))

http_session = requests.Session()


@telegram_on_error(http_session)
@rate_limited(
    event_key="requestContext.authorizer.user",
    prefix=f"pydex-{stage}",
    limit=2,
    period=60,
)
@telegram_on_error(http_session)
def pydex(event, context):
    logger.info("Event: %s", event)

    try:
        body = json.loads(event.get("body"))
    except json.JSONDecodeError as e:
        return build_pydex_response(400, "Invalid JSON body")

    logger.info("Decoded body: %s", body)
    path_parameter = event.get("pathParameters", {})
    request_type = path_parameter.get("request_type")

    logger.info("Request Type: %s", request_type)
    try:
        request = Request(request_type=request_type, data=body.get("data"))
        request.validate()
    except ValueError as e:
        logger.exception(e)
        return build_pydex_response(400, "Invalid request")

    user_id = event.get("requestContext", {}).get("authorizer", {}).get("user")

    input_function = request.data
    instruction = None

    if request_type == "add_docstring":
        instruction = "Add a detailed docstring with arguments, exceptions and return type to the function."
        temperature = 0.2

    if request_type == "add_type_hints":
        instruction = "Add type hints to the function."
        temperature = 0.3

    if request_type == "add_unit_test":
        instruction = "Add an unit test for the function."
        temperature = 0.3

    if request_type == "fix_syntax_error":
        instruction = "Fix the syntax error."
        temperature = 0.0

    if request_type == "improve_code_quality":
        instruction = "Improve code quality."
        temperature = 0.3

    if not instruction:
        return build_pydex_response(400, "Invalid request")

    try:
        with OpenAICodex(user_id=user_id, session=http_session) as codex:
            edition = codex.edit(
                input_=input_function,
                instruction=instruction,
                temperature=temperature,
            )
            return build_pydex_response(200, edition)
    except Exception:
        return build_pydex_error_response(500, "Internal Server Error")


@telegram_on_error(http_session)
def authorizer(event, context):
    logger.info("Event: %s", event)
    resources = find_resources(event, stage)

    token = event.get("authorizationToken")
    if token:
        maybe_user = db.get_user_by_api_token(token)
        if maybe_user:
            allow_policy = create_policy(resources, effect="Allow")
            allow_policy["context"] = {"user": maybe_user.unique_user_id}
            logger.info("Authorization granted: %s", allow_policy)
            return allow_policy

    deny_policy = create_policy(resources, effect="Deny")
    logger.info("Authorization denied: %s", deny_policy)
    return deny_policy
