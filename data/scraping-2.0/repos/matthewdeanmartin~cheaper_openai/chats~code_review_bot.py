"""
Python build script bot.
"""

import asyncio
import traceback

import dotenv
import openai
from openai.types.beta.threads.run import ToolAssistantToolsRetrieval

from chats.ai_filesystem import AIFileSystem
from chats.bots import get_persistent_bot, get_persistent_bot_convo
from chats.chatroom import ChatroomLog

dotenv.load_dotenv()


def get_python_files(base_path):
    import os

    def clean_path(path):
        return path.replace("\\", "/")

    def is_python_file(file):
        return file.endswith(".py")

    python_files = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if is_python_file(file):
                full_path = os.path.join(root, file)
                short_path = full_path.replace(base_path, "").lstrip(os.sep)
                clean_full_path = clean_path(full_path)
                clean_short_path = clean_path(short_path)
                python_files[clean_full_path] = clean_short_path

    return python_files


async def upload_all_files(path) -> str:
    python_files = get_python_files("E:/github/untruncate_json")
    ai_file_system = AIFileSystem()
    for path, file_name in python_files.items():
        with open(path, "rb") as contents:
            try:
                # Skip empties.
                the_bytes = contents.read()
                if len(the_bytes) > 0:
                    results = await ai_file_system.create(file_name, the_bytes, "py")
                    print(results)
            except openai.BadRequestError as bad:
                print(bad)
                raise

    exit()


async def review_code(path) -> str:
    # Just get all the files
    fs = AIFileSystem()
    file_ids = []
    files = []
    for key, value in await fs.list():
        if key == "data":
            for file in value:
                files.append(file)
                file_ids.append(file.id)

    instructions = """You review python code."""
    review_bot = await get_persistent_bot(
        bot_name="Python Code Review bot 3",
        bot_instructions=instructions,
        # Needs to be retrieval capable.
        # 'gpt-4-1106-preview',
        model="gpt-3.5-turbo-1106",
    )
    if False:
        # already done
        await review_bot.enable_file(file_ids)

    review_convo = await get_persistent_bot_convo(review_bot, "")

    chatroom = ChatroomLog("Review Code", review_convo.thread_id)
    chatroom.write_header(review_bot)

    tools = [
        ToolAssistantToolsRetrieval(type="retrieval"),
        # CLI tools? Maybe the "report" style?
        # complexity, etc.
    ]

    # initial_message = f"Review the code in file-XrIZj1ddVDwzFPU2Z2WvWTUG\n" \
    #                   f"First recap it in pseudo code.\n" \
    #                   f"What do you think the code is for?\n" \
    #                   f"Next look for possible bugs.\n"
    initial_message = (
        "Review the code in file-XrIZj1ddVDwzFPU2Z2WvWTUG\n"
        "Do you see any functions here that could be written better, please show me the refactor. Thanks!"
    )
    print(initial_message)
    start_message = await review_convo.add_user_message(initial_message)
    chatroom.add_starting_user_message(start_message)

    # Dance to get a reply from the bot.
    run = await review_convo.create_run(tools=tools)
    run = await review_convo.poll_the_run(run)
    final_message = await review_convo.display_most_recent_bot_message()
    chatroom.add_bot_message(review_bot, final_message)

    final_message_text = final_message.content[0].text.value
    print(final_message_text)
    input("continue?")
    try:
        for file in files:
            if "main" in file.filename:
                continue
            if "version" in file.filename:
                continue
            if "init" in file.filename:
                continue
            print(f"Working on {file}")
            initial_message = (
                f" There should be a handful "
                f"of python files. Please look at the file `{file.filename}` "
                f"and provide suggestions for improvement."
            )
            start_message = await review_convo.add_user_message(initial_message)
            chatroom.add_starting_user_message(start_message)

            # All the bot requests.
            run = await review_convo.create_run(tools=tools)
            run = await review_convo.poll_the_run(run)
            final_message = await review_convo.display_most_recent_bot_message()
            chatroom.add_bot_message(review_bot, final_message)

            final_message_text = final_message.content[0].text.value
            print(final_message_text)
            if input("continue?") != "y":
                break
    except Exception as ex:
        chatroom.add_python_exception(ex, traceback.format_exc())
        print("Failed!")
        raise


if __name__ == "__main__":
    # Python 3.7+
    asyncio.run(review_code("E:/github/untruncate_json"))
    print("Done!")
