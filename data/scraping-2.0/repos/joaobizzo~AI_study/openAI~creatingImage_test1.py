import datetime
from base64 import b64decode
import webbrowser
import openai
from openai.error import InvalidRequestError


def generate_image(prompt, num_image=1, size='512x512', output_format='url'):
    """
    :param prompt: str
    :param num_image: int
    :param size: str
    :param output_format: str
    :return:
    """
    try:
        images = []
        response = openai.Image.create(
            prompt=prompt,
            n=num_image,
            size=size,
            response_format=output_format
        )
        if output_format == 'url':
            for image in response['data']:
                images.append(image.url)
        elif output_format == 'b64_json':
            for image in response['data']:
                images.append(image.b64_json)
        return {'created': datetime.datetime.fromtimestamp(response['created']), 'images': images}
    except InvalidRequestError as e:
        print(e)


API_KEY = # your API key
openai.api_key = API_KEY

SIZES = ('1024x1024', '512x512', '256x256')

prompt = str(input('Input: '))
choice = int(input('Enter 0 to URL and 1 to save as JPG: '))
if choice == 0:
    # URL image
    response = generate_image(prompt, num_image=2, size=SIZES[1])
    timestamp = response['created']
    print(timestamp)
    for image in response['images']:
        webbrowser.open(image)
elif choice == 1:
    # save image
    response = generate_image(prompt, num_image=2, size=SIZES[1], output_format='b64_json')
    prefix = 'demo'
    for index, image in enumerate(response['images']):
        with open(f'{prefix}_{index}.jpg', 'wb') as f:
            f.write(b64decode(image))
