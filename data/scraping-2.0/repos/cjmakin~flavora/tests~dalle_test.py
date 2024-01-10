import psycopg2
import psycopg2.extensions
import openai
import hashlib
import os
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path

openai.api_key = os.getenv('OPENAI_API_KEY')


def getRecipeDescription(id):
    connection = psycopg2.connect(database='flavora_test')
    cursor = connection.cursor()
    query = f'SELECT description FROM recipes WHERE id={id}'

    cursor.execute(query)
    description = cursor.fetchall()

    cursor.close()
    connection.close()

    if description != None:
        return description[0][0]
    else:
        return False


def generateImage(recipe_id, username):
    hashed_username = hashlib.sha256(username.encode('utf-8')).hexdigest()
    recipe_description = getRecipeDescription(recipe_id)

    # Request arguments
    prompt = f'delicious image, no text, of: {recipe_description}'
    size = '512x512'
    n = 1
    response_format = 'b64_json'

    try:
        response = openai.Image.create(
            prompt=prompt, n=n, size=size, user=hashed_username, response_format=response_format)
        data = response['data'][0]["b64_json"]

        return data

    except Exception as e:
        print(e)


def saveImage(data, recipe_id):
    # TODO update path when implementing in django
    file_path = f'{Path().resolve()}/images/recipe_images/{recipe_id}.jpg'

    binary_data = base64.b64decode(data)

    try:
        image = Image.open(BytesIO(binary_data))
        image.save(file_path, "JPEG")
        saved = True
    except Exception as e:
        saved = False
        print(e)

    # Save filepath to DB
    if saved:
        try:
            file_path_escaped = psycopg2.extensions.QuotedString(
                file_path).getquoted().decode()

            connection = psycopg2.connect(database='flavora_test')
            cursor = connection.cursor()

            query = f'UPDATE recipes SET img_path = {file_path_escaped} WHERE id = {recipe_id};'
            cursor.execute(query)

            connection.commit()
            cursor.close()
            connection.close()

            print(
                f'SUCCESS: Image of recipe_id={recipe_id} saved at {file_path}')

        except Exception as e:
            print(e)


data = generateImage(1, 'cjmakin')
saveImage(data, 1)

# for i in range(2, 9):
#     data = generateImage(i, 'terra_liu')
#     saveImage(data, i)
