import json
import requests
from openai import OpenAI
from decouple import config
openai_client = OpenAI(api_key=config("OPENAI_API_KEY"))
huggingfacehub_api_token = config("HUGGINGFACE_API_TOKEN")

def hf_ask(question: str, model_url="https://api-inference.huggingface.co/models/google/flan-t5-xxl") -> str:
    """Ask a question to Huggingface, apply it to every row of a pandas dataframe and return the answer"""
    def pandas_func(row) -> str:
        prompt = question.format(**(dict(row.items())))
        headers = {"Authorization": f"Bearer {huggingfacehub_api_token}"}
        response = requests.post(
            model_url, headers=headers, json={"inputs": prompt})
        if response.status_code != 200:
            return None
        return json.loads(response.content.decode("utf-8"))[0]['generated_text']
    return pandas_func


def chatgpt_ask(question: str, model_name="gpt-3.5-turbo") -> str:
    """Ask a question to chatgpt, apply it to every row of a pandas dataframe and return the answer"""
    def pandas_func(row)-> str:
        try:
            prompt = question.format(**(dict(row.items())))
            completion = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            )
            ret = completion.choices[0].message.content.strip()
            return ret
        except:
            return None
    return pandas_func


def openai_ask_helper(question: str, model_name="text-davinci-003") -> str:
    def pandas_func(row)-> str:
        response = openai_client.completions.create(
        model="text-davinci-003",
        prompt=question.format(**(dict(row.items()))),
        )
        ret = [choice.text.strip().lower() for choice in response.choices]
        return ret
    return pandas_func

def openai_ask_verbelizer(question: str, model_name="gpt-3.5-turbo") -> str:
    """Ask a question to chatgpt, apply it to every row of a pandas dataframe and return the answer"""
    def pandas_func(row)-> str:
        prompt = question.format(**(dict(row.items())))
        completion = openai_yes_or_no(prompt)
        ret = completion
        return ret
    return pandas_func


def openai_yes_or_no(prompts):
    YES_TOKEN = frozenset([5297, 3763, 3363, 8505, 3363, 3763, 43335, 3763, 21560])
    GPT3_YES_TOKEN = frozenset([9642, 14410, 10035, 7566, 14331, 9891])
    NO_TOKEN = frozenset([2949, 645, 1400, 3919, 1400, 645, 15285, 645, 8005])
    GPT3_NO_TOKEN = frozenset([2822, 5782, 912, 2201, 9173, 2360])
    response = openai_client.completions.create(
        model="text-davinci-003",
        prompt=prompts,
        temperature=0,
        logit_bias={t: 100 for t in YES_TOKEN | NO_TOKEN},
        max_tokens=1,
    )
    ret = [choice.text.strip().lower() == "yes" for choice in response.choices]
    return ret


def chat_gpt_ask_functions_multiclass(question: str, categories) -> str:
    def pandas_func(row)-> str:
        prompt = question.format(**(dict(row.items())))
        completion = classify_multiclass(prompt, categories)
        ret = completion
        return ret
    return pandas_func


def classify_multiclass(prompt, categories):
    messages = [{"role": "user", "content": prompt}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "classify_app",
                "description": "Classify to an enum type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "classes": {"type": "array", "items": {"type": "string", "enum": list(categories.index)}},
                    },
                    "required": ["classes"],
                },
            },
        }
    ]
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "classify_app"}}
    )
    ret =  response.choices[0].message.tool_calls[0].function.arguments
    return json.loads(ret)["classes"]


def chat_gpt_ask_functions_most_likley(question: str, categories) -> str:
    def pandas_func(row)-> str:
        prompt = question.format(**(dict(row.items())))
        completion = classify_most_likely(prompt, categories)
        ret = completion
        return ret
    return pandas_func


def classify_most_likely(prompt, categories):
    messages = [{"role": "user", "content": prompt}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "classify_app",
                "description": "Classify to an enum type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "class": {"type": "string", "enum": list(categories.index)},
                    },
                    "required": ["class"],
                },
            },
        }
    ]
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "classify_app"}}
    )
    ret =  response.choices[0].message.tool_calls[0].function.arguments
    return json.loads(ret)["class"]


def jaccard_set(list1, list2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union