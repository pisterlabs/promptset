import openai


def get_completion(prompt, system_msg=None, model="gpt-3.5-turbo"):
    messages = (
        [] if not system_msg else [{"role": "system", "content": system_msg}]
    )
    messages.append({"role": "user", "content": prompt})

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
        )
        reply = response.choices[0].message["content"]
    except openai.error.RateLimitError:
        reply = "I am currently overloaded with other requests."

    return reply
