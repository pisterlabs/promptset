import openai
import os

openai.api_key = api_key = os.getenv("OPENAI_API_KEY")


def ai_function_generation(demo, context, question, requirements, model = "gpt-3.5-turbo"):
    # parse args to comma separated string
    messages = [{"role": "system",
                "content": demo},
                {"role": "user",
                "content": f"Propositions: ```{context}```\nQuestion: ```{question}```, ```{requirements}```"}]

    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = 0
    )

    return response.choices[0].message["content"]

def ai_generation_adjustment(demo, code, error_message, model = "gpt-3.5-turbo"):
    # parse args to comma separated string
    messages = [{"role": "user",
            "content": f"{demo}\n Here is the original code: ```{code}```\n And the exception that was thrown is: ```{error_message}```"}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message["content"]

def ai_generation_check(demo, question, model = "gpt-3.5-turbo"):
    # parse args to comma separated string
    messages = [{"role": "user",
            "content": f"{demo}\n The sentence you are expected to decide is: ```{question}```"}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message["content"]

def ai_function_backconvertion(demo, code, model = "gpt-3.5-turbo"):
    # parse args to comma separated string
    messages = [{"role": "system",
                "content": demo},
                {"role": "user",
                "content": f"Code: ```{code}```"}]

    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = 0
    )

    return response.choices[0].message["content"]

def ai_function_comparison(demo, original, generated, model = "gpt-3.5-turbo"):
    # parse args to comma separated string
    messages = [{"role": "system",
                "content": demo},
                {"role": "user",
                "content": f"Original Propositions: ```{original}```, Generated Propositions: ```{generated}```"}]

    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = 0
    )

    return response.choices[0].message["content"]

def ai_function_regeneration(demo, code, text, model = "gpt-3.5-turbo"):
    # parse args to comma separated string
    messages = [{"role": "system",
                "content": demo},
                {"role": "user",
                "content": f"Original Code: ```{code}```, Problem information: ```{text}```"}]

    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = 0
    )

    return response.choices[0].message["content"]

def ai_function_extraction(demo, text, model = "gpt-3.5-turbo"):
    # parse args to comma separated string
    messages = [{"role": "system",
                    "content": demo},
                {"role": "user",
                    "content": f"Problem information: ```{text}```"}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message["content"]