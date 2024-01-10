# This script is actually used to housekeep the markdown files in this repository.
# See https://github.com/combinatrix-ai/PromptTrail/pull/3 for what it does.

import os

from prompttrail.agent import State
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import LinearTemplate
from prompttrail.agent.templates.openai import (
    OpenAIGenerateTemplate as GenerateTemplate,
)
from prompttrail.agent.templates.openai import OpenAIMessageTemplate as MessageTemplate
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.models.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

templates = LinearTemplate(
    templates=[
        MessageTemplate(
            content="""
You're an AI proofreader that help user to fix markdown.
You're given markdown content by the user.
You only emit the corrected markdown. No explanation, comments, or anything else is needed.
Do not remove > in the code section, which represents the prompt.
""",
            role="system",
        ),
        MessageTemplate(
            content="{{content}}",
            role="user",
        ),
        GenerateTemplate(role="assistant"),
    ],
)

MAX_TOKENS = 8000
MODEL_NAME = "gpt-3.5-turbo-16k"

configuration = OpenAIModelConfiguration(api_key=os.environ.get("OPENAI_API_KEY", ""))
parameter = OpenAIModelParameters(
    model_name=MODEL_NAME, temperature=0.0, max_tokens=MAX_TOKENS
)
model = OpenAIChatCompletionModel(configuration=configuration)

runner = CommandLineRunner(
    model=model,
    parameters=parameter,
    template=templates,
    user_interaction_provider=UserInteractionTextCLIProvider(),
)


def main(
    load_file: str,
):
    # show all log levels
    import logging

    logging.basicConfig(level=logging.INFO)

    load_file_content = open(load_file, "r")
    splits: list[list[str]] = []
    # try splitting by ##

    chunk: list[str] = []
    for line in load_file_content.readlines():
        if line.startswith("## "):
            splits.append(chunk)
            chunk = []
        chunk.append(line)
    if len(chunk) > 0:
        splits.append(chunk)

    corrected_splits: list[str] = []
    for split in splits:
        content = "\n".join(split)
        initial_state = State(data={"content": content})
        state = runner.run(state=initial_state)
        last_message = state.get_last_message()
        content = last_message.content
        print(content)
        corrected_splits.append(content)
    corrected_all = "\n".join(corrected_splits)
    if corrected_all[-1] != "\n":
        corrected_all += "\n"
    save_file_io = open(load_file, "w")
    save_file_io.write(corrected_all)
    save_file_io.close()


if __name__ == "__main__":
    # main(load_file="README.md")
    # main(load_file="CONTRIBUTING.md")
    main(load_file="src/prompttrail/agent/README.md")
