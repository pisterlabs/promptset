from io import BytesIO
from random import randint

from PIL import Image
import pilgram
import openai


def shorten_text(api_token: str, text: str, symbols_count: int = 600, model: str = 'gpt-3.5-turbo-16k'):
    prompt = f'''
    Перескажи нижеследующий текст, сократив до {symbols_count} символов. Пусть текст будет немного большей длины, главное не упустить важные детали.

    {text}
    '''
    openai.api_key = api_token
    completion = openai.ChatCompletion.create(model=model,
                                              messages=[{"role": "assistant", "content": prompt}])
    return completion.choices[0].message.content


def rephrase_title(api_token: str, title: str, model: str = 'gpt-3.5-turbo'):
    prompt = f'Перефразируй  название новости про футбол: {title}. Не используй кавычки при выводе названия.'
    openai.api_key = api_token
    completion = openai.ChatCompletion.create(model=model,
                                              messages=[{"role": "assistant", "content": prompt}])
    return completion.choices[0].message.content


def make_img_unique(input_image: BytesIO,
                    _filter: str = '__original__',
                    pixels_num: int = 100,
                    horizontal_flip: bool = False) -> bytes:
    """
    Make image unique using pixels replacement and instagram filters.

    Pixel replacement code source:
    https://www.geeksforgeeks.org/how-to-manipulate-the-pixel-values-of-an-image-using-python/

    Args:
        input_image(BytesIO): Input image as bytes.
        _filter(string): Instagram filter to be applied.
        pixels_num(int): Number of pixels to be replaced.
        horizontal_flip(bool): Defines if horizontal flip must be applied.

    Returns:
        bytes: Unified image as bytes.

    """
    image = Image.open(input_image)
    width, height = image.size
    for _ in range(pixels_num):

        # get random pixel position
        i, j = randint(1, width - 1), randint(1, height - 1)

        # getting the RGB pixel value.
        r, g, b = image.getpixel((i, j))

        # Apply formula of grayscale:
        grayscale = (0.299 * r + 0.587 * g + 0.114 * b)

        # setting the pixel value.
        image.putpixel((i, j), (int(grayscale), int(grayscale), int(grayscale)))

    if horizontal_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    filtered_img: Image.Image = getattr(pilgram, _filter)(image)
    byte_img = BytesIO()
    filtered_img.save(byte_img, 'PNG')
    return byte_img.getvalue()



