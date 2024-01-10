import argparse
import openai
import json
import os
import time

ADDITIONAL_PROMPT = """
Background: Now, I am using you as a debugging tool.
Rule #1: When debug is required, please only return the revised script.
Rule #2: Please highlight the issue fixing lines between the original and revised using "[REVISED #1]" tag as comments.
Rule #3: When [REVISED #X] tags are already included, please increment them as [REVISED #X+1].
Rule #4: When [REVISED #X] tags are removed, plese keep them as they are; you don't need to add the [REVISED #X] tags for them anymore.
Rule #5: When no revision is added, please not return any code but just return "No revision needed."
Rule #3: However, you shold avoid addiing unnecessary comments on the code; it's just durty.
Rule #4: Your comments are not nesseccary as they are just distructive.
Rule #5: When yes/no question is delivered, please respond it with the Point, Reason, and Example (PRE) method in about three sentences.
"""


def run_gpt(
    api_key,
    engine,
    max_tokens,
    temperature,
    api_type,
    prompt_file,
    history_file,
    N_HISTORY=3,
):
    # Read the prompt
    with open(prompt_file, "r") as f:
        prompt = f.read()

    # Prepend the ADDITIONAL_PROMPT to the user's prompt
    prompt = ADDITIONAL_PROMPT + prompt

    # Load the history
    if api_type == "chat":
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
        else:
            history = []
        history.append({"role": "user", "content": prompt})
    else:
        history = None

    # Set the api key
    openai.api_key = api_key

    # Depending on the api_type use the appropriate API
    if api_type == "chat":
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=engine,
                    messages=history[-N_HISTORY:],
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                )
                break
            except openai.error.InvalidRequestError as e:
                if "context length is" in str(e):
                    # Reduce the history and try again by excluding the first message
                    history = history[1:]
                    print(e)
                else:
                    raise e

            except openai.error.RateLimitError as e:
                print(e)
                time.sleep(3)

        # Update the history with assistant's message
        history.append(
            {
                "role": "assistant",
                "content": response["choices"][0]["message"]["content"],
            }
        )
    else:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )

    # Save the updated history
    if api_type == "chat":
        with open(history_file, "w") as f:
            json.dump(history, f)

    print()
    print("\n" + "=" * 60 + "\n")
    for response in history:
        role = response["role"]
        role = {"user": "YOU", "assistant": "GPT"}[role]
        content = (
            response["content"]
            .replace("Assistant: ", "")
            .replace("assistant: ", "")
            .replace("User: ", "")
            .replace("user: ", "")
        )
        if role == "YOU":
            content.replace("\n\n", "")

        print(role)
        print()
        print(content)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI GPT.")
    parser.add_argument("api_key")
    parser.add_argument("engine")
    parser.add_argument("max_tokens")
    parser.add_argument("temperature")
    parser.add_argument("api_type")
    parser.add_argument("prompt_file")
    parser.add_argument("history_file")
    args = parser.parse_args()

    run_gpt(
        args.api_key,
        args.engine,
        args.max_tokens,
        args.temperature,
        args.api_type,
        args.prompt_file,
        args.history_file,
    )
