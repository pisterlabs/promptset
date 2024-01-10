from typing import Optional
from api import config
from api.user.model import User
from api.util.log import logger
from openai import OpenAI
import json

def get_user_by_username(username: str) -> Optional[User]:
    """
    Return the user model by its username
    If no user found, return None
    """
    # pylint: disable=no-member
    users_result = User.objects(username=username)
    count = users_result.count()
    if count > 1:
        logger.error("%s users matched by username '%s'", count, username)
        raise RuntimeError("Something went wrong")
    if count == 0:
        return None
    return users_result.get()

def get_user_by_id(uid: str) -> Optional[User]:
    """
    Return the user model by its id
    If no user found, return None
    """
    # pylint: disable=no-member
    users_result = User.objects(id=uid)
    print(users_result)
    count = users_result.count()
    if count > 1:
        logger.error("%s users matched by user id '%s'", count, uid)
        raise RuntimeError("Something went wrong")
    if count == 0:
        return None
    return users_result.get()

def calculate_bmi(height, weight):
    # Ensure that height is in meters
    height_in_meters = height / 100
    bmi = weight / (height_in_meters * height_in_meters)

    return bmi

def get_image_info(image_url):
    client = OpenAI(api_key=config.VISION_API_KEY)
    response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
    {
        "role": "user",
        "content": [
        {"type": "text", "text": "Return 5 numbers seperated by commas only, without additional text. I understand that it won't be accurate, but use the image provided to provide your estimates of the following. The first number is the number of grams you think this meal is, \
            the second is estimated calories, the third is estimated fat, the fourth is estimated protien, and the fifth is estimated carbs."},
        {
            "type": "image_url",
            "image_url": {
            "url": image_url,
            # "https://www.watscooking.com/images/dish/large/DSC_1009.JPG"
            "detail": "low"
            },
        },
        ],
    }
    ],
    max_tokens=300,
    )

    response_content = response.choices[0].message.content
    response_content = response_content.replace(" ", "")
    numbers = response_content.split(',')

    # Create a JSON object with keys 'mass', 'cal', 'fat', 'protein', and 'carbs'
    json_object = {
        'weight': numbers[0],
        'calories': numbers[1],
        'fat': numbers[2],
        'proteins': numbers[3],
        'carbs': numbers[4]
}

    json_string = json.dumps(json_object, indent=2)

    print(json_string)
    return json_string
