"""
Chatroom to name a module

User - describes the package
Pypi Name Bot - comes up with names
Pypi Name Judge Bot - evaluates names
"""

import asyncio
import traceback

import dotenv
from openai.types import FunctionDefinition
from openai.types.beta.threads.run import ToolAssistantToolsFunction

from chats.bots import get_persistent_bot, get_persistent_bot_convo
from chats.chatroom import ChatroomLog

dotenv.load_dotenv()

if __name__ == "__main__":

    async def main():
        instructions = """You suggest two to ten pypi packages for people who describe what they need.
- Package should exist and support python 3, as far as we know.
- If we can't think of any, it is okay to say there are not any.
Your answer will follow this format:

Polars and Pandas are good packages for dataframes.
---
1. polars
2. pandas
"""
        name_bot = await get_persistent_bot("Pypi Package Suggestion Bot", instructions, "gpt-3.5-turbo")
        await name_bot.update_instructions(instructions)

        # TODO: UI thing to pick a thread/start new thread
        thread_id = ""
        name_convo = await get_persistent_bot_convo(name_bot, thread_id)

        chatroom = ChatroomLog("Package Suggestion", name_convo.thread_id)
        try:
            chatroom.write_header(name_bot)
            judge_instructions = """
            You judge pypi packages for appropriateness to a users needs. 
            Appropriate packages match the users needs and also exist on pypi."""
            judge_bot = await get_persistent_bot(
                "Pypi package appropriateness judge", judge_instructions, "gpt-3.5-turbo"
            )

            # TODO: Way to track judge thread via chat room, user doesn't specify thread, it has to come from chatroom.
            judge_thread_id = ""
            judge_convo = await get_persistent_bot_convo(judge_bot, judge_thread_id)

            # User's first prompt in a request-response pattern
            # package_description = "I need a package that tells me if a package name is actually module in the stdlib of python."
            package_description = "I want to write a bunch of functions that either interface with grep, or do the same thing that grep does, i.e. search files and folders for text in a variety of ways."

            print(package_description)
            start_message = await name_convo.add_user_message(package_description)
            chatroom.add_starting_user_message(start_message)

            # TODO: message is an uploaded file.

            # Submit user req to agent. Run may involve bot asking to use `functions`
            run = await name_convo.create_run()

            # check run over and over to see if it blew up or finished.
            run = await name_convo.poll_the_run(run)

            # retrieve new messages that appear via threads.messages.list()
            # how do we know they're new?
            # Optionally look at run steps to see if Bot was using code_interpretor, dall-e or bing

            # Show everything. This makes an API call
            bots_ideas_message = await name_convo.display_most_recent_bot_message()
            chatroom.add_bot_message(name_bot, bots_ideas_message)

            ideas = bots_ideas_message.content[0].text.value
            print(ideas)
            # "- The project name must not be  too similar to an existing project and may be confusable. " \
            for_judgement = (
                f"A developer said: `{package_description}`.\n"
                f"These are the suggested packages: ```ideas\n\n{ideas}```\n\n"
                f"What do you think, are these real, are they apt to the task? Please use tools to check for package existence and description."
            )
            print(for_judgement)
            instructions_for_judge = await judge_convo.add_user_message(for_judgement)

            tools = [
                ToolAssistantToolsFunction(
                    function=FunctionDefinition(
                        name="packages_exists",
                        description="Check if a package exists on pypi",
                        parameters={
                            "type": "object",
                            "properties": {
                                "package_names": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                    },
                                    "description": "Names of packages we want to check for existence",
                                },
                            },
                            "required": ["package_names"],
                        },
                    ),
                    type="function",
                ),
                ToolAssistantToolsFunction(
                    function=FunctionDefinition(
                        name="describe_packages",
                        description="Get some descriptive info about packages",
                        parameters={
                            "type": "object",
                            "properties": {
                                "package_names": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                    },
                                    "description": "Names of packages we want descriptive info about",
                                },
                            },
                            "required": ["package_names"],
                        },
                    ),
                    type="function",
                ),
            ]

            # Submit user req to agent. Run may involve bot asking to use `functions`
            judge_run = await judge_convo.create_run(tools=tools)
            run = await judge_convo.poll_the_run(judge_run)
            judgement_response = await judge_convo.display_most_recent_bot_message()
            chatroom.add_bot_message(judge_bot, judgement_response, instructions_for_judge)
            print(judgement_response.content[0].text.value)
        except Exception as ex:
            chatroom.add_python_exception(ex, traceback.format_exc())
            raise

    # Python 3.7+
    asyncio.run(main())
