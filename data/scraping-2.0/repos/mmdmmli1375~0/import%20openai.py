import openai
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import Update

# تنظیم کلید‌های API تلگرام و ChatGPT
openai.api_key = "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"
TELEGRAM_TOKEN = "6929952733:AAGY--Gq9uKviEf-r0N9WgHxyXzhuwR2af8"

# تنظیم کار کننده‌ی تلگرام
def start(update: Update) -> None:
    update.message.reply_text('ربات تلگرام و سرویس ChatGPT فعال است. لطفاً پیامی را برای دریافت پاسخ از سرویس ChatGPT ارسال کنید.')

def handle_message(update: Update) -> None:
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"توضیح مسئله: {update.message.text}\nپاسخ:",
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    reply_text = response.choices[0].text.strip()
    update.message.reply_text(reply_text)

def main() -> None:
    updater = Updater(TELEGRAM_TOKEN)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))