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
        instructions = """You help people pick python package names based on how they describe their package.
- Names should describe the purpose of the package.
- Names should be concise.
If not specified, you'll think up 10. Please answer in a list, e.g. [\n'requests',\n'pandas'\n]
"""
        name_bot = await get_persistent_bot("Pypi Name Bot", instructions)

        # TODO: UI thing to pick a thread/start new thread
        thread_id = ""
        name_convo = await get_persistent_bot_convo(name_bot, thread_id)

        chatroom = ChatroomLog("Pypi Name Bot", name_convo.thread_id)
        try:
            chatroom.write_header(name_bot)
            judge_instrunctions = (
                "Starting with a description of a python package and some names, "
                "you judge if they are good names, pick the best and give some reasons."
            )
            judge_bot = await get_persistent_bot("Pypi Name Judge Bot", judge_instrunctions)

            # TODO: Way to track judge thread via chat room, user doesn't specify thread, it has to come from chatroom.
            judge_thread_id = ""
            judge_convo = await get_persistent_bot_convo(judge_bot, judge_thread_id)

            # User's first prompt in a request-response pattern
            package_description = "A python package will spell check prompts before users submit them to an AI chatbot."

            print(package_description)
            start_message = await name_convo.add_user_message(package_description)
            chatroom.add_starting_user_message(start_message)

            # TODO condense to 1 call
            run = await name_convo.create_run()
            run = await name_convo.poll_the_run(run)
            bots_ideas_message = await name_convo.display_most_recent_bot_message()
            chatroom.add_bot_message(name_bot, bots_ideas_message)

            ideas = bots_ideas_message.content[0].text.value
            print(ideas)
            # "- The project name must not be  too similar to an existing project and may be confusable. " \
            for_judgement = (
                f"The package description: `{package_description}`.\n"
                f"These are the ideas: ```ideas\n{ideas}```\n\n"
                f"What do you think? Please use tools to check for package existence. We can't reuse names."
            )
            print(for_judgement)
            instructions_for_judge = await judge_convo.add_user_message(for_judgement)

            tools = [
                ToolAssistantToolsFunction(
                    function=FunctionDefinition(
                        name="packages_or_variant_exist",
                        description="Check if a package (or close variants) exists on pypi",
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
                        name="package_is_stdlib",
                        description="Check if packages names conflict with stdlib for any version of python",
                        parameters={
                            "type": "object",
                            "properties": {
                                "package_names": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                    },
                                    "description": "Names of packages we want to check for conflict with stdlib",
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
