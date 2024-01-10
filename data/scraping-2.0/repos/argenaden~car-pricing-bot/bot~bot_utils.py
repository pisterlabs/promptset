import requests
import openai
import json
import os
from bot_config import Config

openai.api_key = Config.OPENAI_API_KEY
telegram_bot_token = Config.TELEGRAM_BOT_TOKEN

def load_car_details():
    project_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    json_file_path = os.path.join(project_directory, 'data', 'car_details.json')
    with open(json_file_path, 'r') as file:
        return json.load(file)

car_details = load_car_details()

keywords = {
    "kia": "kia",
    "киа": "киа",
    "hyundai": "hyundai",
    "цены": "price",
    "price": "price",
    "mashina": "car",
    "машина": "car"
}

def generate_answer(question):
    question_lower = question.lower()
    relevant_keywords = [value for key, value in keywords.items() if key in question_lower]

    if relevant_keywords:
        responses = []

        for car_id, car in car_details.items():
            if any(car.get("Производитель", "").lower() == keyword for keyword in relevant_keywords):
                model = car.get("Модель", "Model not specified")
                price = car.get("Цена", "Price not available")
                fuel_type = car.get("Тип Топлива", "Fuel type not specified")
                office_city_state = car.get("Город Офиса", "Office location not specified")
                car_info = f"Model: {model}, Price: {price}, Fuel Type: {fuel_type}, Location: {office_city_state}"

                photo_url = f"https://raw.githubusercontent.com/argenaden/car-pricing-korea/main/car_photos/{car_id}/2.jpg"
                responses.append((photo_url, car_info))

                if len(responses) >= 5:
                    break

        return responses
    else:
        context = "У меня вопрос про автомобили в Корее"
        response = openai.completions.create(
            model="text-davinci-003",
            prompt=f"{context}\nQ: {question}\nA:",
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].text.strip()

def message_parser(message):
    chat_id = message['message']['chat']['id']
    text = message['message']['text']
    return chat_id, text

def send_message_telegram(chat_id, text):
    url = f'https://api.telegram.org/bot{telegram_bot_token}/sendMessage'
    payload = {'chat_id': chat_id, 'text': text}
    return requests.post(url, json=payload)

def send_photo_telegram(chat_id, photo_url, caption):
    url = f'https://api.telegram.org/bot{telegram_bot_token}/sendPhoto'
    payload = {
        'chat_id': chat_id,
        'photo': photo_url,
        'caption': caption
    }
    return requests.post(url, json=payload)

def handle_incoming_message(message):
    chat_id, incoming_question = message_parser(message)
    responses = generate_answer(incoming_question)

    if isinstance(responses, list):
        for photo_url, caption in responses:
            send_photo_telegram(chat_id, photo_url, caption)
    else:
        send_message_telegram(chat_id, responses)
