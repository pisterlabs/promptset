import logging

import openai
import requests

from evals.utils import setup_environment

logger = logging.getLogger(__name__)


_org_ids = {
    "NYU": "org-rRALD2hkdlmLWNVCKk9PG5Xq",
    "FAR": "org-AFgHGbU3MeFr5M5QFwrBET31",
    "ARG": "org-4L2GWAH28buzKOIhEAb3L5aq",
}


def extract_usage(response):
    requests_left = float(response.headers["x-ratelimit-remaining-requests"])
    requests_limit = float(response.headers["x-ratelimit-limit-requests"])
    request_usage = 1 - (requests_left / requests_limit)
    tokens_left = float(response.headers["x-ratelimit-remaining-tokens"])
    tokens_limit = float(response.headers["x-ratelimit-limit-tokens"])
    token_usage = 1 - (tokens_left / tokens_limit)
    overall_usage = max(request_usage, token_usage)
    return overall_usage


def get_ratelimit_usage(data, org_id, endpoint):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
            "OpenAI-Organization": org_id,
        }
        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=20,
        )

        return extract_usage(response)
    except Exception as e:
        logger.warning(f"Error fetching ratelimit usage: {e}")
        return -1


def fetch_ratelimit_usage(org_id, model_name) -> float:
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say 1"}],
    }
    return get_ratelimit_usage(data, org_id, "https://api.openai.com/v1/chat/completions")


def fetch_ratelimit_usage_base(org_id, model_name) -> float:
    data = {"model": model_name, "prompt": "a", "max_tokens": 1}
    return get_ratelimit_usage(data, org_id, "https://api.openai.com/v1/completions")


def get_current_openai_model_usage() -> None:
    models_to_check = [
        "gpt-3.5-turbo-instruct",
        "gpt-4-1106-preview",
        "gpt-4-base",
    ]
    org_names = ["NYU", "FAR", "ARG"]
    result_str = "\nModel usage: 1 is hitting rate limits, 0 is not in use. -1 is error.\n"
    for org in org_names:
        result_str += f"\n{org}:\n"
        for model_name in models_to_check:
            if model_name == "gpt-4-base" or model_name == "gpt-3.5-turbo-instruct":
                if org == "ARG":
                    usage = fetch_ratelimit_usage_base(_org_ids[org], model_name)
                else:
                    continue
            else:
                usage = fetch_ratelimit_usage(_org_ids[org], model_name)
            result_str += f"\t{model_name}:\t{usage:.2f}\n"
        result_str += "\n"
    print(result_str)


if __name__ == "__main__":
    setup_environment()
    get_current_openai_model_usage()
