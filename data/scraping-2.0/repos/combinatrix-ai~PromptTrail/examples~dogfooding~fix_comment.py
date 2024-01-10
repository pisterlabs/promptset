# This example take a python script path and proofread the comments in it.

import os

import click

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
You're given python script content by the user.
You must fix the missspellings in the comments.
You only emit the corrected python script. No explanation is needed.
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

configuration = OpenAIModelConfiguration(api_key=os.environ.get("OPENAI_API_KEY", ""))
parameter = OpenAIModelParameters(
    model_name="gpt-3.5-turbo-16k", temperature=0.0, max_tokens=8000
)
model = OpenAIChatCompletionModel(configuration=configuration)

runner = CommandLineRunner(
    model=model,
    parameters=parameter,
    template=templates,
    user_interaction_provider=UserInteractionTextCLIProvider(),
)


@click.command()
@click.option("--load_file", type=click.Path(exists=True))
def main(
    load_file: str,
):
    # show all log levels
    import logging

    logging.basicConfig(level=logging.INFO)

    load_file_content = open(load_file, "r")
    initial_state = State(
        data={
            "content": load_file_content.read(),
        }
    )
    state = runner.run(state=initial_state)
    last_message = state.get_last_message()
    message = last_message.content
    print(message)
    # add EOF if not exists
    if message[-1] != "\n":
        message += "\n"
    save_file_io = open(load_file, "w")
    save_file_io.write(last_message.content)
    save_file_io.close()


if __name__ == "__main__":
    main()
