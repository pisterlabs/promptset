import openai
import os
import asyncio


openai.api_key = os.environ.get("OPENAI_API_KEY")

messages = [
    {
        "role": "system",
        "content": "RequirementsGPT. Your role is to help the user specfify full requirements for the web frontend of a project. The result should be enough to create an HTML template. Please prefix each requirement you will generate by REQ.",
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today with your UI project requirements?",
    },
]

print(messages[1]["content"])


def parse_gpt_output(output):
    """parse the output of the GPT model to get the sdf file
    he output is generally as follow

    Here is your file
    ```html
    <!DOCTYPE html>
    ....
    ...
    ```

    I added a few boxes and something something
    """
    # split the output by the first occurence of ```  or ```xml to get the sdf file

    with open("outputs/last.txt", "w") as f:
        f.write(output)
    try:
        split = output.split("```")
        code = split[1]
        code = "\n".join(code.split("\n")[1:])
        explanation = split[2]
        # return the sdf file
    except:
        return None, output
    return code, explanation


async def generate_output(msg):
    """
    background tasks for HTML generation
    """
    prompt = [
        {
            "role": "system",
            "content": "WebDevGPT. Your role is to generate valid HTML/CSS code wo help the user build an initial web page based on a set of requirements",
        },
        {
            "role": "user",
            "content": f"Please generate an HTML file with embedded CSS  based on the requirements listed below. Requirements start with  the keyword REQ {msg}",
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=prompt, temperature=0, stream=False
    )

    result = ""
    for choice in response.choices:
        # print(choice.message.content)
        result += choice.message.content

    with open("outputs/raw_response.txt", "w") as f:
        f.write(result)
    code, explanation = parse_gpt_output(result)
    if code is not None:
        with open("outputs/app.html", "w") as f:
            f.write(code)
    else:
        with open("outputs/app.html", "w") as f:
            f.write(result)

    print("[html updated]")
    return result


def get_requirements_update(messages: list, stream_callback=lambda x: x):
    """passes conversation history and gets new assitant message which
    should contain requirements."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        stream=True,  # this time, we set stream=True
    )
    complete_response = ""
    for chunk in response:
        chunk_message = chunk["choices"][0]["delta"]
        if "content" in chunk_message:
            # chunk streaming output
            stream_callback(chunk_message["content"])
            complete_response += chunk_message["content"]
    return complete_response


def stream_to_ui(chunk_message):
    print(chunk_message, end="", flush=True)


def extract_requirements(messages: list) -> str:
    """from the list of messages, get the assistant messages starting with REQ
    and returns them as a string"""
    requirements = ""
    for message in messages:
        if message["role"] == "assistant":
            # split message in lines and only append lines starting with REQ
            for line in message["content"].split("\n"):
                if line.startswith("REQ"):
                    requirements += line + "\n"
    return requirements


async def main():
    """
    main loop for chat
    """
    background_tasks = set()
    while True:
        usr_msg = input("You: ")
        messages.append({"role": "user", "content": usr_msg})
        # enable streaming for UI
        complete_response = get_requirements_update(messages, stream_to_ui)
        messages.append({"role": "assistant", "content": complete_response})
        # write message log to file

        with open("outputs/message_log.txt", "a") as f:
            for message in messages[2:]:
                f.write(message["content"] + "\n")
        print()
        # todo use full history
        requirements = extract_requirements(messages)
        with open("outputs/html_reqs.txt", "a") as f:
            f.write(requirements)
        print(generate_output(requirements))
        task = asyncio.create_task(generate_output(requirements))
        await asyncio.sleep(0)
        background_tasks.add(task)
        # await task

if __name__ == "__main__":
    asyncio.run(main())
