import json
import os
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
from dotenv import load_dotenv
from openai_services import ask_azure_gpt, ask_azure_dalle, load_mock_daily_sales, load_mock_ingredients

app = Flask(__name__)
load_dotenv()

line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))


@app.route("/", methods=["GET"])
def index():
    return "Welcome to OpenAI Line Bot"


@app.route("/message", methods=["GET"])
def message():
    query = request.args.get("q")

    if should_analyze_daily_sales(query):
        daily_sales = load_mock_daily_sales()
        answer = analyze_daily_sales(daily_sales)
        return jsonify({"question": query, "answer": answer})

    if should_analyze_ingredients(query):
        ingredients = load_mock_ingredients()
        answer = analyze_ingredients(ingredients)
        return jsonify({"question": query, "answer": answer})

    if should_generate_image(query):
        image_url = ask_azure_dalle(query)
        return jsonify({"question": query, "image_url": image_url})

    if should_analyze_growth(query):
        daily_sales = load_mock_daily_sales()
        answer = ask_azure_gpt(analyze_growth(daily_sales))
        return jsonify({"question": query, "answer": answer})

    if should_generate_new_menu(query):
        ingredients = load_mock_ingredients()
        answer = ask_azure_gpt(generate_menu(ingredients))
        return jsonify({"question": query, "answer": answer})

    answer = ask_azure_gpt(query)
    return jsonify({"question": query, "answer": answer})


@app.route("/direct", methods=["POST"])
def direct():
    body = request.get_data(as_text=True)
    answer = ask_azure_gpt(body)
    return json.loads(answer)


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        return jsonify({"error": "Invalid signature"}), 400

    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    query = event.message.text

    if should_generate_image(query):
        image_url = ask_azure_dalle(query)
        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text="Here's a sample image:"),
                ImageSendMessage(original_content_url=image_url, preview_image_url=image_url),
            ],
        )
    elif should_analyze_growth(query):
        daily_sales = load_mock_daily_sales()
        answer = ask_azure_gpt(analyze_growth(daily_sales))
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))
    elif should_generate_new_menu(query):
        ingredients = load_mock_ingredients()
        answer = ask_azure_gpt(generate_menu(ingredients))
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))
    elif should_analyze_daily_sales(query):
        daily_sales = load_mock_daily_sales()
        answer = analyze_daily_sales(daily_sales)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))
    elif should_analyze_ingredients(query):
        ingredients = load_mock_ingredients()
        answer = analyze_ingredients(ingredients)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))
    else:
        answer = ask_azure_gpt(query)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))


def analyze_daily_sales(daily_sales):
    date = daily_sales["date"]
    sales = daily_sales["sales"]

    response = f"Sales report for {date}:\n"

    for i, sale in enumerate(sales, start=1):
        item = sale["item"]
        quantity = sale["quantity"]
        revenue = sale["revenue"]
        response += f"{i}. {item}: Sold {quantity} dishes, Revenue: {revenue} baht\n"

    return response


def analyze_ingredients(ingredients):
    response = "Stock summary:\n"
    for ingredient in ingredients["ingredients"]:
        name = ingredient["name"]
        quantity = ingredient["quantity"]
        response += f"- {name}: {quantity} kilograms\n"
    return response


def analyze_growth(daily_sales):
    date = daily_sales["date"]
    sales = daily_sales["sales"]

    response = f"How to make more profits on {date}:\n"

    for i, sale in enumerate(sales, start=1):
        item = sale["item"]
        quantity = sale["quantity"]
        revenue = sale["revenue"]
        response += f"{i}. {item}: Sold {quantity} dishes, Revenue: {revenue} baht\n"

    return response


def generate_menu(ingredients):
    menu = "New menu generated from available ingredients:\n"
    for ingredient in ingredients["ingredients"]:
        name = ingredient["name"]
        quantity = ingredient["quantity"]
        menu += f"- {name}: {quantity} kilograms\n"

    return menu


def should_analyze_daily_sales(query):
    keywords = ["sales report", "summary of sales", "how much did I sell today"]
    query = query.lower()
    for keyword in keywords:
        if keyword in query:
            return True
    return False


def should_analyze_ingredients(query):
    keywords = ["stock summary", "stock status", "remaining ingredients", "how much stock do I have"]
    query = query.lower()
    for keyword in keywords:
        if keyword in query:
            return True
    return False


def should_analyze_growth(query):
    keywords = [
        "how to increase sales",
        "increase profits",
        "how to grow sales",
        "how to increase revenue",
    ]
    query = query.lower()
    for keyword in keywords:
        if keyword in query:
            return True
    return False


def should_generate_new_menu(query):
    trigger_words = [
        "generate menu",
        "create menu",
        "new menu",
        "generate new menu from available ingredients",
        "how to manage remaining ingredients",
    ]
    query = query.lower()
    for word in trigger_words:
        if word in query:
            return True
    return False


def should_generate_image(query):
    trigger_keywords = ["picture", "photo", "image", "รูป", "ภาพ", "รูปภาพ"]
    query = query.lower()
    for phrase in trigger_keywords:
        if phrase in query:
            return True
    return False


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("Starting Flask OpenAI app")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
