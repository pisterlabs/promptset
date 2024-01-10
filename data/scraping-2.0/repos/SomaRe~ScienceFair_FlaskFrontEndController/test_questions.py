import datetime
import random
import requests
from openai import OpenAI
import base64
import requests
from KEYS import OPENAI_API_KEY

def get_three_things():
    three_things = (
    ("car", "tree", "ball"),
    ("pencil", "moon", "shoe"),
    ("apple", "mountain", "watch"),
    ("flower", "book", "fridge"),
    ("guitar", "cloud", "spoon"),
    ("bird", "bottle", "chair"),
    ("lamp", "ocean", "hat"),
    ("computer", "banana", "door"),
    ("glass", "cat", "bridge"),
    ("phone", "sweater", "lake"),
    ("plate", "sky", "beach"),
    ("sandwich", "star", "key"),
    ("dog", "paper", "train"),
    ("pizza", "planet", "ring"),
    ("butterfly", "boot", "tower"),
    ("camera", "lemon", "bed"),
    ("song", "candle", "river"),
    ("grape", "pocket", "museum"),
    ("whistle", "island", "desk"),
    ("chocolate", "kite", "park")
)
    return random.choice(three_things)

def get_reverse_word():
    reverse_words = ("APPLE","BIRD","DOG","FLOWER","GLASS","LAMP","PLATE","SONG","CAMERA","GLASS","PENCIL","CAR")
    return random.choice(reverse_words)

def get_season(month, day):
    if (month == 12 and day >= 21) or (month <= 2 and day <= 19):
        return "Winter"
    elif month <= 5 and day <= 20:
        return "Spring"
    elif month <= 8 and day <= 21:
        return "Summer"
    else:
        return "Fall"
    
def get_date_info():
    current_date = datetime.datetime.now()
    year = current_date.year
    month = current_date.month
    day = current_date.day
    day_of_week = current_date.strftime("%A")

    return {
        "Year": year,
        "Month": month,
        "Day": day,
        "DayOfWeek": day_of_week,
        "Season": get_season(month, day)
    }

def get_date_info_with_season():
    date_info = get_date_info()
    return date_info

def get_image_link():
    image_links = {"wrist watch" : "https://i.pinimg.com/736x/11/13/0a/11130ac9de99eae78af686a9742a15e3.jpg",
                "airplane": 'https://thumbs.dreamstime.com/b/airplane-18327587.jpg',
                "car": 'https://vehicle-images.dealerinspire.com/stock-images/chrome/d51929e056d69529c5bf44c4ceaddf7e.png',
                }
    return random.choice(list(image_links.items()))
    
def get_words_to_click():
    words = ['hello', 'good', 'new', 'happy', 'beautiful']
    # return two random words
    return random.sample(words, 2)

def get_random_time():
    hour = random.randint(1, 12)
    minute = random.choice(range(0, 60, 5))
    return { "hour": hour, "minute": minute }


def get_location_info():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()

        country = data.get("country", "Unknown")
        state = data.get("region", "Unknown")

        return {
            "Country": country,
            "State": state
        }
    except Exception as e:
        return {
            "Country": "Unknown",
            "State": "Unknown",
        }

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_drawing_score(encoded):
    main_image = encode_image("static/image.png")
    client = OpenAI(api_key = OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "First image is a drawing, second image is a actual photo containing 2 pentagons, overlapped on one side, lets call it main image. please tell if the drawing resembeles the main image. score it 1 if all 10 sides present and its overlapping on one side, score it 0 if its not score is 0, no partial score. also please use the this format, {score: 1, description: 'all 10 sides present and its overlapping on one side'} only, follow this json strictly, dont give me any other information, just this json.",
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded}",
            },
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{main_image}",
            },
            },
        ],
        }
    ],
    max_tokens=100,
    )
    print(response.choices[0].message.content)
    # return response.choices[0].message.content


# Example usage:
# location_info = get_location_info()
# print(location_info)


