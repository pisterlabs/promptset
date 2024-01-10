import threading
import time
import re
from typing import List, Dict, Any, Generator, Tuple, Union

from slack_bolt import BoltContext
from slack_sdk import WebClient

from slack.markdown import slack_to_markdown, markdown_to_slack
from slack.slack_ops import update_wip_message

import g4f
from slack.env import AI_MODEL, MAX_TOKENS


def get_answer( *, messages, stream: bool = False, ignored: List[str] = None, ignore_working: bool = False, ignore_stream_and_auth: bool = False, **kwargs ):
    try: 
        for result in g4f.ChatCompletion.create(
            model=g4f.models.default,
            provider=g4f.Provider.Bing,
            messages=messages,
            stream=stream,
            ignored=ignored,
            ignore_working=ignore_working,
            ignore_stream_and_auth=ignore_stream_and_auth,
            **kwargs
        ):
            yield result

    
    except Exception as e:
        # Return error message if an exception occurs
        return (
            f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'
        )    


def write_answer( 
    *,
    client: WebClient,
    wip_reply: dict,
    context: BoltContext,
    user_id: str,
    answer: str,
    messages: List[Dict[str, str]],
    translate_markdown: bool
):
    assistant_reply: Dict[str, str] = {"role": "assistant", "content": f"{answer}"}
    messages.append(assistant_reply)
    threads = []

    try:
        loading_character = " ... :writing_hand:"
        def update_message():
            assistant_reply_text = format_assistant_reply(
                assistant_reply["content"], translate_markdown
            )
            wip_reply["message"]["text"] = assistant_reply_text
            update_wip_message(
                client=client,
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=assistant_reply_text + loading_character,
                messages=messages,
                user=user_id,
            )

        thread = threading.Thread(target=update_message)
        thread.daemon = True
        thread.start()
        threads.append(thread)

        for t in threads:
            try:
                if t.is_alive():
                    t.join()
            except Exception:
                pass

        assistant_reply_text = format_assistant_reply(
            assistant_reply["content"], translate_markdown
        )
        wip_reply["message"]["text"] = assistant_reply_text
        update_wip_message(
            client=client,
            channel=context.channel_id,
            ts=wip_reply["message"]["ts"],
            text=assistant_reply_text,
            messages=messages,
            user=user_id,
        )
    finally:
        for t in threads:
            try:
                if t.is_alive():
                    t.join()
            except Exception:
                pass

def consume_custom_stream_to_write_reply(
    *,
    client: WebClient,
    wip_reply: dict,
    context: BoltContext,
    user_id: str,
    messages: List[Dict[str, str]],
    stream: Generator[Any, Any, None],
    timeout_seconds: int,
    translate_markdown: bool
):
    start_time = time.time()
    assistant_reply: Dict[str, str] = {"role": "assistant", "content": ""}
    messages.append(assistant_reply)
    threads = []
    word_count = 0

    try:
        loading_character = " ... :writing_hand:"
        for chunk in stream:
            spent_seconds = time.time() - start_time
            if timeout_seconds < spent_seconds:
                raise TimeoutError("Stream response timeout")

            # Process the chunk (adapt this part to your specific stream structure)
            assistant_reply["content"] += (chunk)  # Implement process_chunk
            word_count += len(chunk.split())

            # Update message periodically or based on a condition
            if word_count >= 10:

                def update_message():
                    assistant_reply_text = format_assistant_reply(
                        assistant_reply["content"], translate_markdown
                    )
                    wip_reply["message"]["text"] = assistant_reply_text
                    update_wip_message(
                        client=client,
                        channel=context.channel_id,
                        ts=wip_reply["message"]["ts"],
                        text=assistant_reply_text + loading_character,
                        messages=messages,
                        user=user_id,
                    )

                thread = threading.Thread(target=update_message)
                thread.daemon = True
                thread.start()
                threads.append(thread)
                word_count = 0

        # Finalize after stream completion
        assistant_reply_text = format_assistant_reply(
            assistant_reply["content"], translate_markdown
        )
        wip_reply["message"]["text"] = assistant_reply_text
        update_wip_message(
            client=client,
            channel=context.channel_id,
            ts=wip_reply["message"]["ts"],
            text=assistant_reply_text,
            messages=messages,
            user=user_id,
        )
        
    finally:
        # Ensure all threads are joined and the stream is closed
        for t in threads:
            try:
                if t.is_alive():
                    t.join()
            except Exception:
                pass
        try:
            stream.close()
        except Exception:
            pass


# Format message from Slack to send to AI
def format_ai_message_content(content: str, translate_markdown: bool) -> str:
    if content is None:
        return None

    # Unescape &, < and >, since Slack replaces these with their HTML equivalents
    # See also: https://api.slack.com/reference/surfaces/formatting#escaping
    content = content.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    # Convert from Slack mrkdwn to markdown format
    if translate_markdown:
        content = slack_to_markdown(content)

    return content


# Format message from OpenAI to display in Slack
def format_assistant_reply(content: str, translate_markdown: bool) -> str:
    for o, n in [
        # Remove leading newlines
        ("^\n+", ""),
        # Remove prepended Slack user ID
        ("^<@U.*?>\\s?:\\s?", ""),
        # Remove OpenAI syntax tags since Slack doesn't render them in a message
        ("```\\s*[Rr]ust\n", "```\n"),
        ("```\\s*[Rr]uby\n", "```\n"),
        ("```\\s*[Ss]cala\n", "```\n"),
        ("```\\s*[Kk]otlin\n", "```\n"),
        ("```\\s*[Jj]ava\n", "```\n"),
        ("```\\s*[Gg]o\n", "```\n"),
        ("```\\s*[Ss]wift\n", "```\n"),
        ("```\\s*[Oo]objective[Cc]\n", "```\n"),
        ("```\\s*[Cc]\n", "```\n"),
        ("```\\s*[Cc][+][+]\n", "```\n"),
        ("```\\s*[Cc][Pp][Pp]\n", "```\n"),
        ("```\\s*[Cc]sharp\n", "```\n"),
        ("```\\s*[Mm][Aa][Tt][Ll][Aa][Bb]\n", "```\n"),
        ("```\\s*[Jj][Ss][Oo][Nn]\n", "```\n"),
        ("```\\s*[Ll]a[Tt]e[Xx]\n", "```\n"),
        ("```\\s*bash\n", "```\n"),
        ("```\\s*zsh\n", "```\n"),
        ("```\\s*sh\n", "```\n"),
        ("```\\s*[Ss][Qq][Ll]\n", "```\n"),
        ("```\\s*[Pp][Hh][Pp]\n", "```\n"),
        ("```\\s*[Pp][Ee][Rr][Ll]\n", "```\n"),
        ("```\\s*[Jj]ava[Ss]cript\n", "```\n"),
        ("```\\s*[Ty]ype[Ss]cript\n", "```\n"),
        ("```\\s*[Pp]ython\n", "```\n"),
    ]:
        content = re.sub(o, n, content)

    # Convert from OpenAI markdown to Slack mrkdwn format
    if translate_markdown:
        content = markdown_to_slack(content)

    return content


def build_system_text(
    system_text_template: str, translate_markdown: bool, context: BoltContext
):
    system_text = system_text_template.format(bot_user_id=context.bot_user_id)
    # Translate format hint in system prompt
    if translate_markdown is True:
        system_text = slack_to_markdown(system_text)
    return system_text

