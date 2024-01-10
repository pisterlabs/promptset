from .log import logger_group
from logbook import Logger
import simplematrixbotlib as botlib
import nio
import markdown
import mistletoe
from . import db
import openai
import threading
import time

logger = Logger("bot.commands")
logger_group.add_logger(logger)

async def help(room_id: str, bot: botlib.Bot, command: str):
    if command == 'info':
        await _send_message(
            room_id,
            bot,
            f"""
The `info` command returns some information about the bot.
Usage:
```plaintext
info
```
""",
            f"""
The <code>info</code> command returns some information about the bot.<br>
Usage:<br>
<pre><code>info
</code></pre>
""")
    elif command == 'list':
        await _send_message(
            room_id,
            bot,
            f"""
The `list` command lists the current sessions in this room.
Usage:
```plaintext
list
```
""",
            f"""
The <code>list</code> command lists the current sessions in this room.<br>
Usage:<br>
<pre><code>list
</code></pre>
""")
    elif command == 'delete':
        await _send_message(
            room_id,
            bot,
            f"""
The `delete` command deletes a session from this room.
Usage:
```plaintext
delete (session-hash)
```
""",
            f"""
The <code>delete</code> command deletes a session from this room.<br>
Usage:<br>
<pre><code>delete (session-hash)
</code></pre>
""")
    elif command == 'active':
        await _send_message(
            room_id,
            bot,
            f"""
The `active` command returns the hash of the currently active session in this room.
Usage:
```plaintext
active
```
""",
            f"""
The <code>active</code> command returns the hash of the currently active session in this room.<br>
Usage:<br>
<pre><code>active
</code></pre>
""")
    elif command == 'activate':
        await _send_message(
            room_id,
            bot,
            f"""
The `activate` command selectes the session to activate/use.
Usage:
```plaintext
activate (session-hash)
```
""",
            f"""
The <code>activate</code> command selectes the session to activate/use.<br>
Usage:<br>
<pre><code>activate (session-hash)
</code></pre>
""")
    elif command == 'new':
        await _send_message(
            room_id,
            bot,
            f"""
The `new` command starts a new session with the given prompt.
Usage:
```plaintext
new (prompt...)
```
""",
            f"""
The <code>new</code> command starts a new session with the given prompt.<br>
Usage:<br>
<pre><code>new (prompt...)
</code></pre>
""")
    else:
        await _send_message(
            room_id,
            bot,
            f"""
To see some information about this bot, use the `info` command.
This `help` command provides usage information for all the available commands.
Usage:
```plaintext
help [command]
```
Here is a list of commands:
|command|description|
|---|---|
|`help`|Returns usage information about commands.|
|`info`|Returns some information about the bot.|
|`list`|Lists the current sessions in this room.|
|`delete`|Deletes a session from this room.|
|`active`|Returns information about currently active session in this room.|
|`activate`|Activates a session for this room.|
|`new`|Creates a new session starting with the prompt following the command.|
""",
            f"""
To see some information about this bot, use the <code>info</code> command.<br>
This <code>help</code> command provides usage information for all the available commands.<br>
Usage:<br>
<pre><code>help [command]
</code></pre>
Here is a list of commands:<br>
<table>
    <thead>
        <tr>
            <th>command</th><th>description</th>
        </tr>
    </thead>
    <tbody>
        <tr><td><code>help</code></td><td>Returns usage information about commands.</td></tr>
        <tr><td><code>info</code></td><td>Returns some information about the bot.</td></tr>
        <tr><td><code>list</code></td><td>Lists the current sessions in this room.</td></tr>
        <tr><td><code>delete</code></td><td>Deletes a session from this room.</td></tr>
        <tr><td><code>active</code></td><td>Returns information about currently active session in this room.</td></tr>
        <tr><td><code>activate</code></td><td>Activates a session for this room.</td></tr>
        <tr><td><code>new</code></td><td>Creates a new session starting with the prompt following the command.</td></tr>
    </tbody>
</table>
""")

async def info(room_id: str, bot: botlib.Bot):
    await bot.api.send_markdown_message(
        room_id,
        f"""
This bot was created by Domi.
It brings the functionality of ChatGPT into any matrix room.
You are seeing this text because you used `info`.
To interact with the bot, you have to mention it!
To see a list of available commands, use `help`.
""",
        "m.text"
        )

async def list(room_id: str, bot: botlib.Bot):
    result = db.get_session_list(room_id)
    if result:
        md_table = ''
        html_table = ''
        for row in result:
            md_table += f"|`{row[0]}`|{row[1]}|{row[2]}|\n"
            html_table += f"<tr><td><code>{row[0]}</code></td><td>{row[1]}</td><td>{row[2]}</td></tr>"
        await _send_message(
            room_id,
            bot,
            f"""
Here is a list of the current sessions in this room:
|session hash|description|timestamp|
|---|---|---|
{md_table}
""",
            f"""
Here is a list of the current sessions in this room:<br>
<table>
    <thead>
        <tr>
            <th>session hash</th><th>description</th><th>timestamp</th>
        </tr>
    </thead>
    <tbody>
        {html_table}
    </tbody>
</table>
""")
    else:
        await _send_message(
            room_id,
            bot,
            f"There are currently no sessions in this room...",
            f"There are currently no sessions in this room...")

async def delete(room_id: str, bot: botlib.Bot, hash_part: str):
    if len(hash_part) > 32:
        await bot.api.send_markdown_message(room_id, "Specified hash part too long (should be under 33 chars).")
        return
    # Check if there are multiple session beginning with that hash part
    result = db.get_sessions_by_hash(room_id, hash_part)
    if len(result) > 1:
        md_table = ''
        html_table = ''
        for row in result:
            md_table += f"|`{row[0]}`|{row[1]}|{row[2]}|\n"
            html_table += f"<tr><td><code>{row[0]}</code></td><td>{row[1]}</td><td>{row[2]}</td></tr>"
        await _send_message(
            room_id,
            bot,
            f"""
Could not delete specified session because there are multiple sessions starting with the same hash part.
Please specify more digits of the hash.
Here is a list of the sessions starting with the specified hash part:
|session hash|description|timestamp|
|---|---|---|
{md_table}
""",
            f"""
Could not delete specified session because there are multiple sessions starting with the same hash part.<br>
Please specify more digits of the hash.<br>
Here is a list of the sessions starting with the specified hash part:<br>
<table>
    <thead>
        <tr>
            <th>session hash</th><th>description</th><th>timestamp</th>
        </tr>
    </thead>
    <tbody>
        {html_table}
    </tbody>
</table>
""")
    else:
        # delete the session
        result = db.delete_session(room_id, hash_part)
        if result[1]:
            await _send_message(
            room_id,
            bot,
            f"""
Deleted session specified by hash.
Activated session with hash `{result[1][0]}`.
Session description:
> {result[1][1]}
""",
            f"""
Deleted session specified by hash.<br>
Activated session with hash <code>{result[1][0]}</code>.<br>
Session description:<br>
<blockquote>{result[1][1]}</blockquote>
""")
        else:
            await bot.api.send_markdown_message(
                room_id,
                "Deleted session specified by hash. No session activated because it was the last session of the room.")

async def active(room_id: str, bot: botlib.Bot):
    result = db.get_active_session(room_id)
    if result:
        await _send_message(
            room_id,
            bot,
            f"""
Active session has hash `{result[0]}`.
Session description:
> {result[1]}
""",
            f"""
Active session has hash <code>{result[0]}</code>.<br>
Session description:<br>
<blockquote>{result[1]}</blockquote>
""")
    else:
        await bot.api.send_markdown_message(
            room_id,
            "No active session because room contains no sessions.")

async def activate(room_id: str, bot: botlib.Bot, hash_part: str):
    if len(hash_part) > 32:
        await bot.api.send_markdown_message(room_id, "Specified hash part too long (should be under 33 chars).")
        return
    # Check if there are multiple session beginning with that hash part
    result = db.get_sessions_by_hash(room_id, hash_part)
    if len(result) > 1:
        md_table = ''
        html_table = ''
        for row in result:
            md_table += f"|`{row[0]}`|{row[1]}|{row[2]}|\n"
            html_table += f"<tr><td><code>{row[0]}</code></td><td>{row[1]}</td><td>{row[2]}</td></tr>"
        await _send_message(
            room_id,
            bot,
            f"""
Could not activate specified session because there are multiple sessions starting with the same hash part.
Please specify more digits of the hash.
Here is a list of the sessions starting with the specified hash part:
|session hash|description|timestamp|
|---|---|---|
{md_table}
""",
            f"""
Could not activate specified session because there are multiple sessions starting with the same hash part.<br>
Please specify more digits of the hash.<br>
Here is a list of the sessions starting with the specified hash part:<br>
<table>
    <thead>
        <tr>
            <th>session hash</th><th>description</th><th>timestamp</th>
        </tr>
    </thead>
    <tbody>
        {html_table}
    </tbody>
</table>
""")
    else:
        # delete the session
        result = db.activate_session(room_id, hash_part)
        if result[1]:
            await _send_message(
            room_id,
            bot,
            f"""
Activated session with hash `{result[1][0]}`.
Session description:
> {result[1][1]}
""",
            f"""
Activated session with hash <code>{result[1][0]}</code>.<br>
Session description:<br>
<blockquote>{result[1][1]}</blockquote>
""")
        else:
            await bot.api.send_markdown_message(
                room_id,
                "No session activated because there are no sessions in the room.")

async def generic(room_id: str, bot: botlib.Bot, message: str, new=False):

    db.create_room_if_not_exists(room_id)

    raw_conversation = db.get_or_create_conversation(room_id, message, 10, new=new)

    if raw_conversation[0][0] == 0:
        # last messag from assisstand. not possible.
        raise Exception("At this point, the last message should be from the user...")

    messages = []

    for i in range(len(raw_conversation) - 1, -1, -1):
        messages.append(
            {
                "role": "user" if raw_conversation[i][0] == 1 else "assistant",
                "content": raw_conversation[i][1]
            })

    result = [None]
    req_thread = threading.Thread(target=ai_api_request, args=(messages, result))
    req_thread.start()

    while req_thread.is_alive():
        await bot.api.async_client.room_typing(
            room_id, True, 500
        )

    await bot.api.async_client.room_typing(room_id, False)

    resp = result[0]

    if resp:
        resp = resp.get("choices")[0].get("message")
        if resp:
            content = resp.get("content")
            db.create_new_message(room_id, content, False)
            await bot.api.send_markdown_message(room_id, mistletoe.markdown(content))

def ai_api_request(messages, result):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages,
        temperature = 0,
    )
    result[0] = response

async def new_error(room_id: str, bot: botlib.Bot):
    await bot.api.send_markdown_message(room_id, "Could not create new session. Please provide a prompt.")
    return

async def _room_send(room_id: str, bot: botlib.Bot, content: dict):

    # CAUTION Does not support encryption...

    resp = await bot.api.async_client.room_send(
        room_id,
        "m.room.message", # messagetype
        content=content,
        ignore_unverified_devices=bot.config.ignore_unverified_devices
    )
    return resp

async def _send_message(room_id: str, bot: botlib.Bot, message: str, formatted_message: str):

    # CAUTION Does not support encryption...

    resp = await _room_send(
        room_id,
        bot,
        {
            "msgtype": "m.text",
            "body": message,
            "format": "org.matrix.custom.html",
            "formatted_body": formatted_message
        }
    )
    if isinstance(resp, nio.RoomSendResponse):
        return resp.event_id

async def _edit_message(room_id: str, bot: botlib.Bot, event_id: str, message: str, formatted_message: str):

    # CAUTION Does not support encryption...

    resp = await _room_send(
        room_id,
        bot,
        {
            "m.new_content": {
            "msgtype": "m.text",
            "body": message,
            "format": "org.matrix.custom.html",
            "formatted_body": markdown.markdown(message, extensions=['nl2br'])
            },
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": event_id
            },
            "msgtype": "m.text",
            "body": message,
            "format": "org.matrix.custom.html",
            "formatted_body": formatted_message
        }
    )
    if isinstance(resp, nio.RoomSendResponse):
        return resp.event_id

def _prepend_start(message: str) -> str:
    return " * " + message

def _markdownify(message: str) -> str:
    return markdown.markdown(message, extensions=['nl2br'])