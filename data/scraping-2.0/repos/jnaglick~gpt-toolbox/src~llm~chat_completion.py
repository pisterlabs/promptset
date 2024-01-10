import openai

from utils import console, env

from .count_tokens import count_tokens
from .model_specs import get_model_spec, ModelType

def setup():
    openai_api_key = env["OPENAI_API_KEY"]
    if not openai_api_key:
        raise ValueError("Put your OpenAI API key in the OPENAI_API_KEY environment variable.")
    openai.api_key = openai_api_key

setup()

def compose_system(system):
    return [{
        "role": "system",
        "content": system
    }]

def compose_examples(examples):
    # TODO experiment with 'name' field (https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb)
    out = [] # TODO use list comprehension
    for example in examples:
        out.append({"role": "user", "content": example[0]})
        out.append({"role": "assistant", "content": example[1]})
    return out

def compose_user(user):
    return [{
        "role": "user",
        "content": user
    }]

def compose_messages(system, examples, user):
    return [
        *compose_system(system),
        *compose_examples(examples),
        *compose_user(user),
    ]

def chat_completion_token_counts(system, examples, user, model: ModelType):
    return {
        "system": count_tokens(compose_system(system), model, count_priming_tokens=False),
        "examples": count_tokens(compose_examples(examples), model, count_priming_tokens=False),
        "user": count_tokens(compose_user(user), model, count_priming_tokens=False),
        "total_prompt": count_tokens(compose_messages(system, examples, user), model),
        "model_max": get_model_spec(model)["max_tokens"]
    }

def chat_completion(system, examples, user, model=ModelType.GPT_4):
    try:
        model_spec = get_model_spec(model)

        messages = compose_messages(system, examples, user)

        completion = openai.ChatCompletion.create(
            model=model_spec["id"],
            messages=messages,
            temperature=0, # based on HuggingGPT
        )

        return completion
    except openai.error.APIError as e:
        console.error(f"(llm) OpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
        console.error(f"(llm) Failed to connect to OpenAI API: {e}")
        pass
    except openai.error.RateLimitError as e:
        console.error(f"(llm) OpenAI API request exceeded rate limit: {e}")
        pass
    except Exception as e:
        console.error(f"(llm) {e.__class__.__name__}: {e}")
        pass
