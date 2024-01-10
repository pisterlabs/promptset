import asyncio
import datetime
import glob
import json
import os
import aiofiles
from openai.error import APIConnectionError, InvalidRequestError
from lib.ApiBuilder import ApiBuilder
from lib.Log import Log
from lib.default_config import default_config

messages: list[dict] = [{"role": "system", "content": "You are a helpful assistant."}]
current_file_name: str = ""
events_loop = asyncio.get_event_loop()


def generateCatalogue(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def getChatLogsPath():
    doc_path = os.path.join(os.path.expanduser("~"), "Documents")
    return os.path.join(doc_path, "chat_logs")


def getCurrentFileName():
    global current_file_name
    fileName = datetime.datetime.now().strftime(
        "%Y-%m-%d-%H-%M-%S") + ".json" if current_file_name == "" else current_file_name
    current_file_name = fileName
    return fileName


async def openCurrentFileBy(mode):
    fileName = getCurrentFileName()
    filePath = getChatLogsPath()
    file = await aiofiles.open(os.path.join(filePath, fileName), mode=mode)
    return file


async def write(msg: list[dict], fp):
    for item in msg:
        data2json = json.dumps(item) + '\n'
        await fp.write(data2json)


async def save():
    global messages, current_file_name
    fileName = getCurrentFileName()
    Log.info("Saving File {}".format(fileName))

    fp = await openCurrentFileBy('w')
    await write(messages, fp)
    await fp.close()
    Log.info("Saved File {}".format(fileName))


async def append(msg: list[dict]):
    global current_file_name

    fp = await openCurrentFileBy('a')
    await write(msg, fp)
    await fp.close()


# noinspection PyTypeChecker
async def setCurrentFile(fileName: str):
    global current_file_name, messages
    current_file_name = fileName
    filePath = getChatLogsPath()
    file = os.path.join(filePath, fileName)
    if not os.path.exists(file):
        fp = await openCurrentFileBy('w')
        await fp.write(json.dumps(messages[0]) + '\n')
        await fp.close()
        messages = messages[0:1]
    else:
        fp = await openCurrentFileBy('r')
        messages.clear()
        async for line in fp:
            item = json.loads(line)
            messages.append(item)
        await fp.close()


async def allChats():
    filePath = getChatLogsPath()
    Log.point("Existing Files：")
    for file in glob.glob(os.path.join(filePath, "*.json")):
        Log.answer(file)


async def selectChat(fileName):
    global messages
    if not fileName.endswith('.json'):
        fileName += '.json'
    try:
        Log.info("Loading File {}".format(fileName))
        await setCurrentFile(fileName)
        Log.info("Loaded File {}, Messages Is {}".format(fileName, messages))
    except FileNotFoundError:
        Log.error("File Does Not Exist", "FileNotFoundError")


def parseResult(completions):
    result = completions.choices[0]["message"]["content"]
    messages.append({"role": "assistant", "content": result})
    Log.answer(result)
    Log.info("Finish")


def parseResult_stream(completions):
    result = ""
    for chunk in completions:
        chunk_message = chunk['choices'][0]['delta']
        res = chunk_message.get('content', '')
        Log.answer(res, end='')
        result += res
    Log.answer('\n')
    Log.info("Finish")
    messages.append({"role": "assistant", "content": result})


def printChatLog(msg: list[dict]):
    for m in msg:
        if m.get("role") == "user":
            print(m.get("content"))
        elif m.get("role") == "assistant":
            Log.answer(m.get("content"))


def not_auto_modify_cons():
    send_messages = messages[-default_config.max_context_size:]
    Log.info("Current context's size is {}".format(len(send_messages)))

    res = ApiBuilder.ChatCompletion(send_messages)
    parseResult_stream(res) if default_config.chatCompletionConfig.stream else parseResult(res)
    asyncio.create_task(append(messages[-2:]))


def auto_modify_cons():
    send_messages = messages[-default_config.max_context_size:]
    Log.info("Current context's size is {}".format(len(send_messages)))
    res = ApiBuilder.ChatCompletion(send_messages)
    parseResult_stream(res) if default_config.chatCompletionConfig.stream else parseResult(res)

    if default_config.max_context_size - len(messages) < 0.3 * default_config.max_context_size:
        default_config.max_context_size = default_config.max_context_size + 2

    asyncio.create_task(append(messages[-2:]))


async def chat():
    global messages
    filePath = getChatLogsPath()
    fileName = getCurrentFileName()
    Log.info("File Name: {}".format(os.path.join(filePath, fileName)))
    Log.point("Start Chatting")

    printChatLog(messages)

    need_input = True
    while True:
        if need_input:
            content = await asyncio.get_running_loop().run_in_executor(None, input, '')
            if content == "quit":
                await save()
                break
            messages.append({"role": "user", "content": content})
        need_input = True

        try:
            if not default_config.auto_modify_cons:
                not_auto_modify_cons()
            else:
                auto_modify_cons()
        except InvalidRequestError:
            if default_config.max_context_size > 1 and default_config.auto_modify_cons:
                default_config.max_context_size = int(default_config.max_context_size / 2)
                Log.warn("The number of tokens exceeds the limit. Automatically reduce the context size and "
                         "prepare to resend the request.")
                need_input = False
            else:
                Log.error("Possibly because the input token exceeds the maximum limit",
                          "InvalidRequestError")
        except APIConnectionError or TimeoutError:
            Log.error("Connection timed out. Please check the network or try again later",
                      "APIConnectionError")


async def image(prompt):
    try:
        Log.answer(ApiBuilder.Image(prompt)["data"][0]["url"])
    except APIConnectionError or TimeoutError:
        Log.error("Connection timed out. Please check the network or try again later", "APIConnectionError")
    except InvalidRequestError:
        Log.error("Unable to understand prompt", "InvalidRequestError")


async def translate(prompt):
    try:
        Log.answer(ApiBuilder.Transcriptions(prompt)["text"])
    except APIConnectionError or TimeoutError:
        Log.error("Connection timed out. Please check the network or try again later", "APIConnectionError")
    except InvalidRequestError:
        Log.error("Possibly because the input token exceeds the maximum limit", "InvalidRequestError")
    except FileNotFoundError:
        Log.error("File Not Found", "FileNotFoundError")


Log.info("File Directory Generating")
chat_logs_path = getChatLogsPath()
generateCatalogue(getChatLogsPath())
Log.info("File Directory Generated. Located at {}".format(chat_logs_path))

__all__ = [
    "save",
    "selectChat",
    "allChats",
    "chat",
    "image",
    "translate"
]
