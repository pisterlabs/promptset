"""
Copyright start
MIT License
Copyright (c) 2023 Fortinet Inc
Copyright end
"""
# from sys import api_version
# from urllib import response
import openai
import arrow
import re
from requests_toolbelt.utils import dump
from bs4 import BeautifulSoup
from jsonschema import validate
from connectors.core.connector import get_logger, ConnectorError
from .constants import *
import tiktoken
import requests

logger = get_logger(LOGGER_NAME)


# logger.setLevel(logging.DEBUG)


def _validate_json_schema(_instance, _schema):
    try:
        validate(instance=_instance, schema=_schema)
        return _instance
    except Exception as err:
        logger.error("Error: {0} {1}".format(SCHEMA_ERROR, err))
        raise ConnectorError("Error: {0} {1}".format(SCHEMA_ERROR, err))


def _remove_html_tags(text):
    tag_stripped = BeautifulSoup(text, "html.parser").text
    return re.sub(r'@\w+\s', '', tag_stripped)


def _build_messages(params):
    ''' builds the message list based on the chat type '''
    operation = params.get('operation')
    messages = [
        {
            "role": "system",
            "content": "Be concise and helpful assistant."
        }
    ]
    if operation == 'chat_completions':
        messages.append({"role": "user", "content": _remove_html_tags(params.get('message'))})
    elif operation == 'chat_conversation':
        replies = _validate_json_schema(params.get('messages'), MESSAGES_SCHEMA)
        for message in replies:
            message.update({'content': _remove_html_tags(message['content'])})
        messages = messages + replies
    return messages


def __init_openai(config):
    openai.api_key = config.get('apiKey')
    openai_args = {"key": config.get('apiKey')}
    api_type = config.get("api_type")
    if api_type:
        openai.api_type = "azure"
        openai.base_url = config.get("api_base")
        openai.api_version = config.get("api_version")
        openai_args.update({
            "base_url": config.get("api_base"),
            "api_type": "azure",
            "api_version": config.get("api_version")
        })
    return openai_args


def chat_completions(config, params):
    __init_openai(config)
    model = params.get('model')
    if not model:
        model = 'gpt-3.5-turbo'
    temperature = params.get('temperature')
    top_p = params.get('top_p')
    max_tokens = params.get('max_tokens')
    messages = _build_messages(params)
    logger.debug("Messages: {}".format(messages))
    openai_args = {"model": model, "messages": messages}
    other_fields = params.get('other_fields', {})
    if config.get("deployment_id"):
        openai_args.update({"deployment_id": config.get("deployment_id")})
    if temperature:
        openai_args.update({"temperature": temperature})
    if max_tokens:
        openai_args.update({"max_tokens": max_tokens})
    if top_p:
        openai_args.update({"top_p": top_p})
    if other_fields:
        openai_args.update(other_fields)
    openai_args['timeout'] = params.get('timeout') if params.get('timeout') else 600
    return openai.chat.completions.create(**openai_args).model_dump()


def list_models(config, params):
    __init_openai(config)
    return openai.models.list().model_dump()


def get_usage(config, params):
    date = arrow.get(params.get('date', arrow.now().int_timestamp)).format('YYYY-MM-DD')
    query_param = {'date': date}
    api_type = config.get("api_type")
    if api_type:
        base_url = config.get("api_base").strip("/")
        if base_url.startswith('http') or base_url.startswith('https'):
            url = "{0}/openai/deployments/{1}/usage".format(base_url, config.get('deployment_id'))
        else:
            url = "https://{0}/openai/deployments/{1}/usage".format(base_url, config.get('deployment_id'))
        query_param["api-version"] = config.get('api_version')
    else:
        url = USAGE_URL
    response = make_rest_call(config, url=url, params=query_param)
    return response


def count_tokens(config, params):
    """Returns the number of tokens in a text string."""
    input_text = params.get("input_text")
    model = params.get("model")
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(input_text))
    return {"tokens": num_tokens}


def check(config):
    try:
        list_models(config, {})
        return True
    except Exception as err:
        logger.error('{0}'.format(err))
        if err.error:
            raise ConnectorError(err.error.get("message"))
        raise ConnectorError('{0}'.format(err))


def make_rest_call(config, url, method='GET', **kwargs):
    try:
        headers = {
            "Authorization": "Bearer {0}".format(config.get('apiKey'))
        }
        try:
            from connectors.debug_utils.curl_script import make_curl
            debug_headers = headers.copy() if headers else None
            debug_headers["Authorization"] = "*****************"
            make_curl(method=method, url=url, headers=debug_headers, **kwargs)
        except Exception as err:
            logger.info("Error: {0}".format(err))
        response = requests.request(method=method, url=url, headers=headers, **kwargs)
        if response.ok:
            return response.json()
        else:
            try:
                logger.error("Error: {0}".format(response.json()))
                raise ConnectorError('Error: {0}'.format(response.json()))
            except Exception as error:
                raise ConnectorError('{0}'.format(response.text if response.text else str(response)))
    except requests.exceptions.SSLError as e:
        logger.exception('{0}'.format(e))
        raise ConnectorError('{0}'.format(e))
    except requests.exceptions.ConnectionError as e:
        logger.exception('{0}'.format(e))
        raise ConnectorError('{0}'.format(e))
    except Exception as e:
        logger.error('{0}'.format(e))
        raise ConnectorError('{0}'.format(e))
