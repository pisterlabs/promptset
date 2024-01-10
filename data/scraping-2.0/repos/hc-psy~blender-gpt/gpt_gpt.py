from openai import OpenAI
import re
from .gpt_cst import SYSTEMPROMPTS


def post_process(final_txt):
    final_txt = re.findall(
        r'```(.*?)```', final_txt, re.DOTALL)[0]

    final_txt = re.sub(
        r'^python', '', final_txt, flags=re.MULTILINE)

    return final_txt


def chatgpt(context, api_key=''):

    if not api_key:
        raise Exception("Please provide an OpenAI API key")

    scene = context.scene

    # sysprompt preparation
    messages = [{"role": "system", "content": system_prompt}
                for system_prompt in SYSTEMPROMPTS]

    # add previous messages
    for msg in scene.history[-8:]:
        if msg.type == "GPT":
            messages.append(
                {"role": "assistant", "content": "```\n" + msg.content + "\n```"})
        else:
            messages.append({"role": "user", "content": msg.content})

    # add the current user message
    if messages[-1]["role"] != "user":
        formatted_message = f"Please provide me with Blender (3D software) python code satisfying the following task: {scene.prompt_input}. \n. Do not provide with anything that is not Python code. Do not provide explanations and comments."
        messages.append({"role": "user", "content": formatted_message})

    # send message to GPT
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=scene.model,
        messages=messages,
        temperature=scene.creativity,
    )

    try:
        final_txt = response.choices[0].message.content
        return post_process(final_txt)
    except IndexError:
        return ''
