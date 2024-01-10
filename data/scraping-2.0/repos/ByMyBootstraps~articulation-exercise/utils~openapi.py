import openai
from openai import APIConnectionError, APIError, Timeout

from utils.decorators import usecache

from utils.spend import recordSpending
@usecache(
    valueSourceFn=lambda *args, **kwargs: f"chat",
    cacheKeyFn=lambda *args, **kwargs: f"{args}",
    useCacheFn=lambda *args, **kwargs: kwargs.get("use_cache", True)
)
def chat( messages, model, max_tokens, n, auto_retry=True):
    try:
        resp = openai.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            n=n,
        )
    except (APIConnectionError, APIError, Timeout) as e:
        if auto_retry:
            print( "APIConnectionError: retrying..." )
            return chat( messages, model, auto_retry=False )
        else:
            raise e
    except Exception as e:
        print( "IF THIS SHOULD HAVE AUTO-RETRIED, EDIT THIS CODE")
        raise e
    
    recordSpending( resp.model, resp.usage.prompt_tokens, resp.usage.completion_tokens )

    # needs to be json serializable since they changed the api...
    resp = {
        "choices": [
            {
                "message": {
                    "content": choice.message.content
                }
            }
            for choice in resp.choices
        ],
        "model": resp.model,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens
        }
    }

    return resp    