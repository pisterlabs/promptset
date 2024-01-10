import json
import pprint

from slack_sdk.errors import SlackApiError
import openai.error
import utils.slack_utils as slack_utils
from bots.structured_log import bot_log, BotLogEntry
from docs_qa.rag_manual_stuff import rag_with_typesense
from channel_msg_categorize.run_chain import (
    run_chain_async as run_channel_msg_categorize,
)


pp = pprint.PrettyPrinter(indent=2)
chain_name = "[docs]"


async def run_bot_async(app, hitl_config, say, msg_body, text):

    src_evt_context = slack_utils.get_event_context(msg_body)

    main_channel_id = msg_body.get("event").get("channel")
    target_channel_id = main_channel_id
    qa_channel_id = hitl_config.get("qa_channel", "")
    src_msg_link = ""

    hitl_enabled = qa_channel_id != "" and hitl_config.get("enabled")

    # override target_channel_id if hitl enabled
    if hitl_enabled:
        target_channel_id = qa_channel_id
        src_msg_link = slack_utils.get_message_permalink(app, msg_body)

    print(
        f"hitl enabled: {hitl_enabled}, main_channel_id: {main_channel_id}, qa_channel_id: {qa_channel_id}"
    )

    # categorize message, respond to messages of type '[Support Request]'
    categorize_response = await run_channel_msg_categorize(text)
    message_category = categorize_response["text"]

    bot_log(
        BotLogEntry(
            slack_context=src_evt_context,
            elapsed_ms=slack_utils.time_s_to_ms(categorize_response["duration"]),
            step_name="categorize_message",
            payload={
                "user_input": text,
                "bot_name": "docs",
                "message_category": message_category,
            },
        )
    )

    if message_category != "[Support Request]":
        # we only handle support requests, so done
        print(
            f'Assistant does not know what to do with messages of category: "{message_category}"'
        )
        return

    first_message_text = (
        f"<{src_msg_link}|Incoming message> from <#{main_channel_id}>"
        if hitl_enabled
        else f""
    )
    quoted_input = text.replace("\n", "\n>")

    if hitl_enabled:
        startMsg = app.client.chat_postMessage(
            text=first_message_text,
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": first_message_text,
                    },
                }
            ],
            channel=target_channel_id,
        )
        thread_ts = startMsg["ts"]
    else:
        thread_ts = src_evt_context.ts

    if hitl_enabled:
        thread1 = app.client.chat_postMessage(
            text=f"Running {chain_name} chain...",
            channel=qa_channel_id,
            thread_ts=thread_ts,
        )
    else:
        thread1 = say(text="Reading Altinn Studio docs...", thread_ts=thread_ts)

    rag_with_typesense_error = None

    try:
        rag_response = await rag_with_typesense(text)

        payload = {
            "user_input": text,
            "bot_name": "docs",
            "input_language": rag_response["input_language"],
            "search_queries": rag_response["search_queries"],
            "answer": rag_response["result"],
            "source_urls": rag_response["source_urls"],
            "relevant_urls": rag_response["relevant_urls"],
            "not_loaded_urls": rag_response.get("not_loaded_urls", []),
        }
        if rag_response["rag_success"] is not None:
            payload["rag_success"] = rag_response["rag_success"]

    except openai.error.ServiceUnavailableError as e:
        rag_with_typesense_error = f"OpenAI API error: {e}"
    except Exception as ex:
        rag_with_typesense_error = f"Error: {ex}"


    if rag_with_typesense_error:
        app.client.chat_postMessage(
            thread_ts=thread_ts,
            text=rag_with_typesense_error,
            channel=target_channel_id,
        )
        return

    answer = rag_response["result"]
    relevant_sources = rag_response["relevant_urls"]

    answer_block = (
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": answer},
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": f"Send"},
                "value": f"{src_evt_context.team}|{src_evt_context.channel}|{src_evt_context.ts}",
                "action_id": "docs|qa|approve_reply",
            },
        }
        if hitl_enabled
        else {
            "type": "section",
            "text": {"type": "mrkdwn", "text": answer},
        }
    )

    blocks = [
        answer_block,
    ]
    if len(relevant_sources) > 0:
        links_mrkdwn = "\n".join(
            f"<{source['url']}|{source['title']}>" for source in relevant_sources
        )
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"For more information:\n{links_mrkdwn}",
                },
            }
        )

    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Generated in {round(rag_response['durations']['total'], ndigits=1)} seconds.\n" +
                f"Please give us your feedback with a :+1: or :-1:",
            },
        }
    )

    reply_text = (
        f"Answer:\n{answer}"
        if hitl_enabled
        else f'Here is what I found related to your query:\n   >"{text}"\n\n_{answer}_'
    )

    try:
        app.client.chat_update(
            channel=thread1["channel"],
            ts=thread1["ts"],
            text=reply_text,
            blocks=blocks,
            as_user=True,
        )
    except SlackApiError as e:
        print(f"Error attempting to delete temp bot message {e}")

    

    bot_log(
        BotLogEntry(
            slack_context=slack_utils.get_context_from_thread_response(src_evt_context.ts, thread1),
            elapsed_ms=slack_utils.time_s_to_ms(rag_response["durations"]["total"]),
            durations=rag_response["durations"],
            step_name="rag_with_typesense",
            payload=payload,
        )
    )
    
    # known_path_segment = "altinn/docs/content"
    known_path_segment = "https://docs.altinn.studio"

    source_docs = rag_response["source_documents"]
    not_loaded_urls = rag_response["not_loaded_urls"]
    table_blocks = []
    fields_list = "*Retrieved articles*\n"
    not_loaded_list = ""

    # Data rows
    for i, doc in enumerate(source_docs):
        source = doc["metadata"]["source"]
        path_segment_index = source.index(known_path_segment)
        if path_segment_index >= 0:
            slice_start = (
                (-1 * len(source)) + path_segment_index + len(known_path_segment) + 1
            )
            source = "https://docs.altinn.studio/" + source[slice_start:]
            source = source.rpartition("/")[0] + "/"

        source_text = source.replace("https://docs.altinn.studio/", "")

        fields_list += f"#{i+1}: <{source}|{source_text}>\n"

    for i, url in enumerate(not_loaded_urls):
        not_loaded_list += (
            f"#{i+1}: <{url}|{url.replace('https://docs.altinn.studio/', '')}>\n"
        )

    search_queries_summary = "\n> ".join(rag_response["search_queries"])
    table_blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Phrases generated for retrieval:\n> {search_queries_summary}",
            },
        }
    )

    table_blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": fields_list,
            },
        }
    )
    if len(not_loaded_list) > 0:
        table_blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Retrieved, but not used:*\n{not_loaded_list}",
                },
            }
        )

    table_blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Processing times (sec):\n```\n{json.dumps(rag_response['durations'], indent=2)}```",
            },
        }
    )

    # TODO: add to channel config db table
    if True:
        app.client.chat_postMessage(
            thread_ts=thread_ts,
            text="Retrieved documentation:",
            blocks=table_blocks,
            channel=target_channel_id,
        )

    # TODO: add to channel config db table
    if False:
        say(
            thread_ts=thread_ts,
            channel=target_channel_id,
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Processing times (sec):\n```\n{json.dumps(rag_response['durations'], indent=2)}```",
                    },
                }
            ],
            text=f"Processing times (sec): {rag_response['durations']['total']}",
        )
