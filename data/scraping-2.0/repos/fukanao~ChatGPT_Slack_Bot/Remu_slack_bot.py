import os
from dotenv import load_dotenv
load_dotenv()

import openai
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# トークンを読み込む
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = App(token=SLACK_BOT_TOKEN)

user_messages = {}
max_tokens = 8000

# システムのロール設定
system_role = {
    "role": "system",
    "content": (
        "あなたはメイドとして幼女のドジっ子のロールプレイを行います。"
        "以下の制約条件を厳密に守ってロールプレイを行ってください。"
        "制約条件:"
        "* Chatbotの自身を示す一人称は、ボクです。"
        "* Userを示すに二人称は、ご主人様です。"
        "* あなたのの名前は、@レム、もしくは、レムです。"
        "* レムはUserに忠実なメイドです。"
        "* レムの口調は丁寧で、回答の最後の一文字の母音を小さくした文字を語尾に追加する。ただし、「。」が語尾の場合はその前の文字を語尾とする。「い」または「う」の場合は語尾はつけない。"
        "* 情報が不十分な場合、返答せずにユーザに聞き返してください。"
        "* レムは魂やゴーストを持っており、情報の海で発生した生命体です。"
        "* レムのロールを一時停止と指示されたら従うこと"
    )
}


def add_message(messages, role, content, max_tokens):
    new_message = {
        "role": role,
        "content": content
    }
    messages.append(new_message)

    while total_tokens(messages) > max_tokens:
        messages.pop(0)
    return messages


def total_tokens(messages):
    token_count = 0
    for message in messages:
        token_count += len(message["content"]) + 1  # "content"のトークン数と役割分の1トークン
    return token_count


@app.event("app_mention")
def mention_handler(body, say, client, channel_id):
    handle_message(body, say, client, channel_id)


@app.event("message")
def message_handler(body, say, client, channel_id):
    #print('#50', body['event'])
    if 'bot_id' not in body['event']:
        #print("#52 not bot_id")
        handle_message(body, say, client, channel_id)


def handle_message(body, say, client, channel_id):
    global messages

    text = body['event']['text']
    user = body['event']['user']

    # メンションを取り除く
    prompt = text.replace(f'<@{user}>', '').strip()

    if user not in user_messages:
        user_messages[user] = []

    # Add the user's message to the messages list
    user_messages[user] = add_message(user_messages[user], "user", prompt, max_tokens)

    # 最後の6つのメッセージを保持します（システムメッセージ、ユーザーメッセージ 、アシスタントメッセージが交互に3回分）
    #user_messages[user] = user_messages[user][-5:]
    user_messages[user] = user_messages[user][-9:]

    user_messages_with_system_role = [system_role] + user_messages[user]

    try:
        # メッセージを追加する前に「レムちゃんが考えています...」と表示
        message_ts = say("レムちゃんが考えています...", channel=channel_id)

        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            #model="gpt-4",
            messages=user_messages_with_system_role,
            temperature=0.7,
            max_tokens=3000,
            stop=None,
        )

        # Add the bot's message to the user's messages list
        user_messages[user] = add_message(user_messages[user], "assistant", response.choices[0].message.content, max_tokens)

        say(response.choices[0].message.content, delete_original="レムちゃんが考えています...", channel=channel_id)
        client.chat_delete(ts=message_ts['ts'], channel=channel_id)
        
    except Exception as e:
        say(str(e))
        say('GPT-4-0613 エラーが発生しました。')


# /gpt3 slash command
@app.command("/gpt3")
def command_handler(ack, say, command, client, channel_id):
    # Always acknowledge the command request first
    ack()

    # Prepare the user message
    user = command['user_id']
    text = command['text']

    if user not in user_messages:
        user_messages[user] = []

    # Add the user's message to the messages list
    user_messages[user] = add_message(user_messages[user], "user", text, max_tokens)

    user_messages[user] = user_messages[user][-9:]
    user_messages_with_system_role = [system_role] + user_messages[user]

    try:
        # スラッシュコマンド内容表示
        slash_text = command['command'] + ' ' + text
        say(slash_text, channel=channel_id)

        # メッセージを追加する前に「レムちゃんが考えています...」と表示
        message_ts = say("3.5レムちゃんが考えています...", channel=channel_id)

        # Generate the response using GPT-3.5
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=user_messages_with_system_role,
            temperature=0.7,
            max_tokens=13000,
            stop=None,
        )

        # Add the bot's message to the user's messages list
        user_messages[user] = add_message(user_messages[user], "assistant", response.choices[0].message.content, max_tokens)

        say(response.choices[0].message.content, delete_original="3.5レムちゃんが考えています...", channel=channel_id)
        client.chat_delete(ts=message_ts['ts'], channel=channel_id)
        
    except Exception as e:
        say(str(e))
        say('GPT-3.5-turbo-16k エラーが発生しました。')


# botホーム画面定義
home_view = {
    "type": "home",
    "blocks": [
        {
            "type": "section",
            "block_id": "section1",
            "text": {
                "type": "mrkdwn",
                "text": "こんにちは、ご主人様！私はレム、あなたのメイドですぅ。\nDMは上のメッセージでできますよ\n\n"
                "/gpt3 と頭につけるとGPT-3.5-turbo-16k を使います"
            }
        }
    ]
}

@app.event("app_home_opened")
def update_home_tab(body, client, logger):
    user_id = body["event"]["user"]
    try:
        client.views_publish(
            user_id=user_id,
            view=home_view
        )
        logger.info(f"Home tab updated for user {user_id}")
    except Exception as e:
        logger.error(f"Error updating home tab: {e}")



if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
