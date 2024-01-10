from app.models.weather_type import WeatherType
from app.models.product import Product
from openai import OpenAI
from dotenv import load_dotenv
import random

load_dotenv()
client = OpenAI()

def generate_openai_advertisement(prompt:str, max_tokens=100) -> str:
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            model="gpt-3.5-turbo",
            max_tokens=max_tokens
        ).choices[0].message.content
    except Exception as e:
        print(e)
        return None

    return response


def generate_advertisement(weather_id:int, lang:str = 'en', max_tokens:int = 30) -> tuple[dict[str, str], int]:
    weather_type = WeatherType.query.get(weather_id)

    if not weather_type:
        return {'error': 'Invalid Weather Type ID'}, 400

    products = Product.query.filter_by(weather_type_id=weather_id).all()

    if not products:
        return {'error': 'No product found for this weather'}, 404

    chosen_product = random.choice(products)
    if lang == 'en':
        prompt = f"Introducing our product named '{chosen_product.name}' for {weather_type.type_name} weather! Make an advertisement of maximum {max_tokens} tokens"
    else:
        prompt = f"Rédige une publicité accrocheuse du produit '{chosen_product.name}' adapté à un climat {weather_type.type_name} de maximum {max_tokens} tokens"

    advertisement_text = generate_openai_advertisement(prompt, max_tokens)
    if advertisement_text == None:
        advertisement_text = chosen_product.name

    return chosen_product.name, advertisement_text
