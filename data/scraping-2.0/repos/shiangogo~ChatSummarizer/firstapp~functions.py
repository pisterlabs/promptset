from django.conf import settings

from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextSendMessage

from datetime import datetime, timedelta
from firstapp.models import Message
import openai

line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)

openai.api_key = settings.OPENAI_API_KEY

def message_event_to_object(event, is_in_group, summarize_request = False):
    message_obj = Message()
    if is_in_group:
        message_obj.group_id = event.source.group_id
        message_obj.group_name = line_bot_api.get_group_summary(event.source.group_id).group_name
        message_obj.user_name = line_bot_api.get_group_member_profile(event.source.group_id, event.source.user_id).display_name
    else:
        message_obj.group_id = None
        message_obj.group_name = None
        message_obj.user_name = line_bot_api.get_profile(event.source.user_id).display_name

    message_obj.id = int(event.message.id)
    message_obj.user_id = event.source.user_id
    message_obj.sent_at = datetime.fromtimestamp(int(event.timestamp) / 1000.0)
    message_obj.unsent_at = None

    if event.message.type == "text" and not summarize_request:
        message_obj.message = event.message.text
    elif event.message.type == "text":
        message_obj.message = f"（{message_obj.user_name}向AI要求重點整理對話）"
    elif event.message.type == "sticker":
        sticker_keywords = ", ".join(event.message.keywords)
        message_obj.message = f"（傳送了一個{sticker_keywords}的貼圖）"
    elif event.message.type == "image":
        message_obj.message = "（傳送了一張圖片）"
    else:
        message_obj.message = None

    return message_obj


def parse_prompt_into_dict(text):
    split_text = text.split()
    if split_text[0] != "總結":
        return False
    elif len(split_text) >= 3 and split_text[0] == "總結" and split_text[1].isdigit():
        days = split_text[1]
        keywords = "、".join(split_text[2:])
    elif len(split_text) == 2 and split_text[0] == "總結" and split_text[1].isdigit():
        days = split_text[1]
        keywords = None
    else:
        days = 1
        keywords = "、".join(split_text[1:])
    return {"days":int(days), "keywords": keywords}

def fetch_data_from_message_table(group_id, user_id, days):
    start_date = datetime.now().date() - timedelta(days=int(days))
    if group_id:
        data = Message.objects.filter(group_id=group_id, sent_at__gte=start_date, unsent_at__isnull=True)
        return data

def ask_ai_for_summarization(chat, keywords = None, model = settings.AI_MODEL):
    if keywords:
        prompt = f"請重點整理以下有關{keywords}的對話，\n{chat}\n如果沒有{keywords}相關的對話，你就簡短地說沒有相關對話。"
    else:
        prompt = f"幫我重點整理以下對話，\n{chat}"
    return openai.ChatCompletion.create(
        model = model,
        messages = [
            { "role": "system", "content": "Assistant helps users summarize their conversation and reply in traditional Chinese." },
            { "role": "user", "content": prompt }
        ]
    )["choices"][0]["message"]["content"]