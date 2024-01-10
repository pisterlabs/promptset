from functools import lru_cache
from pathlib import Path

from cookiecutter import main
from jinja2.ext import Extension

cookiecutter_json_path = Path(__file__).parent / "cookiecutter.json"
def hooked_dump(f):
    if getattr(f, 'is_monkeypatched', False):
        return f

    def wrapper(replay_dir, template_name: str, context: dict):
        print(context)
        post_input_context_hook(context["cookiecutter"], prompts=load_prompts())
        return f(replay_dir=replay_dir, template_name=template_name, context=context)

    wrapper.is_monkeypatched = True
    wrapper._orig = f
    return wrapper


main.dump = hooked_dump(main.dump)


class ContextModifyExtension(Extension):
    pass

@lru_cache()
def load_prompts():
    import json
    with open(cookiecutter_json_path) as f:
        return json.load(f)["__prompts__"]

def post_input_context_hook(context, prompts):
    for k, v in list(context.items()):
        if k.startswith("__"):
            context[k[2:]] = context[k]

    empty_keys = []

    for k, v in context.items():
        if not v:
            empty_keys.append(k)


    for keyname, instructions in prompts.items():
        if not context.get(keyname):
            context[keyname] = generate_context_value(
                key_name=keyname,
                instructions=instructions,
                cookie_context=context
            )

    if context.get("python_packages"):
        python_package_list = context["python_packages"].split(",")
        python_package_list = [p.strip() for p in python_package_list]
        context["python_package_list"] = python_package_list

def filtered_cookie_context(context):
    keys_to_filter = {
        "_extensions", "_output_dir", "_repo_dir", "_checkout"

    }
    cookie_context = {}
    for k, v in context.items():
        if k not in keys_to_filter:
            cookie_context[k] = v
    return cookie_context


def generate_context_value(key_name, instructions, cookie_context):
    import openai
    print(f"Generating value for '{key_name}'...")
    cookie_context = filtered_cookie_context(cookie_context)
    template = (
        f"Below is JSON for a cookiecutter python project generation.  Generate a value for the `{key_name}` field.\n"
        f"{instructions}\n"
        f"```\n{cookie_context}\n```\n\n"
        f"{key_name}:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": template
            },
        ],
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    response_message = response["choices"][0]["message"]
    content = response_message["content"]
    print(f"{content}\n")
    return content.strip().strip('"').strip("'")
