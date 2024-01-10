"""
Re-usable, one-shot bot for summarizing and compressing text.
"""

import asyncio
import traceback

import dotenv
from openai.types import FunctionDefinition
from openai.types.beta.threads.run import ToolAssistantToolsFunction

from chats.bots import get_persistent_bot, get_persistent_bot_convo
from chats.chatroom import ChatroomLog
from chats.tool_code.text_shorteners import LOTS_OF_TEXT, count_tokens, readability_scores

dotenv.load_dotenv()


async def shorten_this_text(text: str, target_model: str, goal_tokens: int, kind_of_text: str) -> str:
    current_token_count = count_tokens(text)
    if current_token_count <= goal_tokens:
        return text
    scores = readability_scores(text)
    scores_text = (
        f"Current text's readability scores: Flesch-Kincaid: {scores['flesch_kincaid']}, "
        f"Coleman-Liau: {scores['coleman_liau']}, Gunning Fog: {scores['gunning_fog']}. "
    )

    instructions = """You help summarize and shorten texts."""
    short_bot = await get_persistent_bot("Text summarization and shortening bot", instructions)
    short_convo = await get_persistent_bot_convo(short_bot, "")

    chatroom = ChatroomLog("Shorten Text", short_convo.thread_id)
    try:
        chatroom.write_header(short_bot)

        start_message = await short_convo.add_user_message(
            f"{scores_text} Make the following text shorter, currently {current_token_count} tokens, "
            f"shortened to {goal_tokens} tokens, that's about 1 page or less while preserving readability. You must check token count!"
            f"Don't include these instructions in the shortened text!\n\n{text}"
        )
        chatroom.add_starting_user_message(start_message)

        tools = [
            ToolAssistantToolsFunction(
                function=FunctionDefinition(
                    name="count_tokens",
                    description="Count tokens",
                    parameters={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to count tokens in",
                            },
                        },
                        "required": ["text"],
                    },
                ),
                type="function",
            ),
            ToolAssistantToolsFunction(
                function=FunctionDefinition(
                    name="readability_scores",
                    description="Get some readability scores",
                    parameters={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to calculate readability scores for",
                            },
                        },
                        "required": ["text"],
                    },
                ),
                type="function",
            ),
            ToolAssistantToolsFunction(
                function=FunctionDefinition(
                    name="word_count",
                    description="Count words ignoring whitespace and punctuation",
                    parameters={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to calculate word count for",
                            },
                        },
                        "required": ["text"],
                    },
                ),
                type="function",
            ),
        ]
        # Submit user req to agent. Run may involve bot asking to use `functions`
        run = await short_convo.create_run(tools=tools)

        # check run over and over to see if it blew up or finished.
        run = await short_convo.poll_the_run(run)

        # Show everything. This makes an API call
        final_message = await short_convo.display_most_recent_bot_message()
        chatroom.add_bot_message(short_bot, final_message)

        text = final_message.content[0].text.value
        current_token_count = count_tokens(text)
        print(f"Current token count: {current_token_count}")
        scores = readability_scores(text)
        scores_text = (
            f"Current text's readability scores: Flesch-Kincaid: {scores['flesch_kincaid']}, "
            f"Coleman-Liau: {scores['coleman_liau']}, Gunning Fog: {scores['gunning_fog']}. "
        )
        print(scores_text)
    except Exception as ex:
        chatroom.add_python_exception(ex, traceback.format_exc())
        print("Failed!")
        raise


if __name__ == "__main__":
    # Python 3.7+
    asyncio.run(shorten_this_text(LOTS_OF_TEXT, "uh....", 700, "txt"))
    print("Done!")
