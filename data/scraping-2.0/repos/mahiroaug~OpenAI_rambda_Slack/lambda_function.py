import openai
import os
import json
import re
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
openai.organization = os.environ["OPENAI_ORGANIZATION"]
openai.api_key = os.environ["OPENAI_API_KEY"]
OPENAI_GPT_MODEL = os.environ["OPENAI_GPT_MODEL"]
OPENAI_PRICE_INPUT = os.environ["OPENAI_PRICE_INPUT"]
OPENAI_PRICE_OUTPUT = os.environ["OPENAI_PRICE_OUTPUT"]
CHAT_GPT_SYSTEM_PROMPT = """
You are an excellent AI assistant Slack Bot.
Please output your response message according to following format.

- bold: "*bold*"
- italic: "_italic_"
- strikethrough: "~strikethrough~"
- code: " \`code\` "
- link: "<https://slack.com|link text>"
- block: "\`\`\` code block \`\`\`"
- bulleted list: "* item1"

Be sure to include a space before and after the single quote in the sentence.
ex) word\`code\`word -> word \`code\` word

If the question is Japanese, please answer in Japanese.

Let's begin.
"""

def lambda_handler(event, context):
    
    print("event: ", event)
    if "x-slack-retry-num" in event["headers"]:
        return {"statusCode": 200, "body": json.dumps({"message": "No need to resend"})}

    body = json.loads(event["body"])
    text = re.sub(r"<@.*>", "", body["event"]["text"])
    channel = body["event"]["channel"]
    thread_ts = body["event"].get("thread_ts") or body["event"]["ts"]
    print("input: ", text, "channel: ", channel, "thread:", thread_ts)

    # get thread messages
    thread_messages_response = slack_client.conversations_replies(channel=channel, ts=thread_ts)
    messages = thread_messages_response["messages"]
    messages.sort(key=lambda x: float(x["ts"]))
    #print("messages:",messages)

    # get recent 30 messages in the thread
    prev_messages = [
        {
            "role": "assistant" if "bot_id" in m and m["bot_id"] else "user",
            "content": re.sub(r"<@.*>|`info: prompt.*USD\)` ", "", m["text"]),
        }
        for m in messages[0:][-30:]
    ]
    print("prev_messages:",prev_messages)

    # Create_completion
    openai_response = create_completion(prev_messages,text)
    print("openaiResponse: ", openai_response)
    tkn_pro = openai_response["usage"]["prompt_tokens"]
    tkn_com = openai_response["usage"]["completion_tokens"]
    tkn_tot = openai_response["usage"]["total_tokens"]
    cost_pro = tkn_pro * float(OPENAI_PRICE_INPUT) / 1000
    cost_com = tkn_com * float(OPENAI_PRICE_OUTPUT) / 1000
    cost = cost_pro + cost_com
    msg_head = "\n `info: prompt + completion = %s + %s = %s tokens(%.4f USD)` " % (tkn_pro,tkn_com,tkn_tot,cost)
    res_text = openai_response["choices"][0]["message"]["content"]
    ##answer = msg_head + res_text
    answer = res_text + msg_head
    print("answer:",answer)
    post_message(channel, answer, thread_ts)

    return {"statusCode": 200, "body": json.dumps({"message": openai_response})}




def create_completion(prev_msg,new_text):
    model=OPENAI_GPT_MODEL
    prompt=[
        {
            "role": "system",
            "content": CHAT_GPT_SYSTEM_PROMPT
        },
        *prev_msg,
        {
            "role": "user",
            "content": new_text
        },
    ]
    print("mdoel:",model,"prompt:",prompt)
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt
        )
        #print("openaiResponse: ", response)
        return response
    except Exception as err:
        print("Error: ", err)

def post_message(channel, text, thread_ts):
    try:
        response = slack_client.chat_postMessage(
            channel=channel,
            text=text,
            as_user=True,
            thread_ts=thread_ts,
            reply_broadcast=True
        )
        print("slackResponse: ", response)
    except SlackApiError as e:
        print("Error posting message: {}".format(e))