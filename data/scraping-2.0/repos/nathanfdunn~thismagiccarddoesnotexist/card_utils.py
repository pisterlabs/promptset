
from typing import Literal, Optional
import openai
import json
import os
import replicate
# from browser_utils import WebsiteAutomation
import cloudinary.uploader
import cloudinary
from mtg_design_api import render_mtg_card

cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

openai.api_key = os.getenv('OPENAI_API_KEY')


def legacy_for_schema(
        card_name: str = 'Missing', 
        mana_cost: str = '',
        rules_text: str = '',
        card_type: Literal['Artifact', 'Creature', 'Land', "Instant", 'Sorcery', 'Enchantment', 'Planeswalker'] = 'Artifact',
        flavor_text: str = '',
        rarity: Literal['Common', 'Uncommon', 'Rare', 'Mythic Rare'] = 'Common',
        power: int = 0,
        toughness: int = 0,
        art_description: str = '',
        explanation: str = '',
        ) -> str:
    """
    Creates the image for a Magic: The Gathering card with the provided details.

    :param card_name: The name of the card.
    :param mana_cost: The mana cost of the card.
    :param rules_text: Describes any triggered, activated, static abilities the card has if applicable.
    :param card_type: The type of the card (e.g., Artifact, Creature, Land, Instant, Sorcery, Enchantment, Planeswalker).
    :param flavor_text: The flavor text of the card.
    :param rarity: The rarity of the card (e.g., Common, Uncommon, Rare, Mythic Rare).
    :param power: The power of the card (relevant for Creature type cards).
    :param toughness: The toughness of the card (relevant for Creature type cards).
    :param art_description: The description of the art on the card.
    :param explanation: Explanation for how the card satisfies the prompt.
    """

# def create_card_image(card: MTGCard, temp_dir: str):
#     temp_dir = tempfile.mkdtemp()
#     final_img_local_path = render_mtg_card(
#         temp_dir=temp_dir,
#         card_name=card_name, 
#         mana_cost=mana_cost,
#         rules_text=rules_text,
#         card_type=card_type,
#         flavor_text=flavor_text,
#         rarity=rarity,
#         power=power,
#         toughness=toughness,
#         explanation=explanation,
#         art_url=art_url
#     )

    
#     upload_result = cloudinary.uploader.upload(final_img_local_path)
#     final = upload_result['url']
#     return final

    # payload = {
    #     'card-title': card_name,
    #     'mana-cost': mana_cost,
    #     'type': card_type,
    #     'rarity': rarity,
    #     'rules-text': rules_text,
    #     'flavor-text': flavor_text,
    #     'artwork': art_url,
    #     'artist': 'Stable Diffusion',
    #     'power': str(power),
    #     'toughness': str(toughness),
    #     'designer': 'Zilth'
    # }

    # print('calling with payload', payload)
    # site = WebsiteAutomation()
    # site.login()
    # result = site.submit_form(payload)
    # print('created this card', result)
    # return result

from art_generator import correct_aspect_ratio

def get_art_url(art_description, temp_dir):
    try:
        return get_bing_url(art_description, temp_dir)
        # return get_dalle_url(art_description)
    except Exception as e:
        import logging
        logging.exception('Looks like we might have tripped the safety checker')
        return None

import BingImageCreator
import sys
import tempfile
import pathlib
from PIL import Image

def get_bing_url(art_description, temp_dir):
    original = sys.argv
    # with tempfile.gettempdir() as temp_dir:
    print('downloading bing to', temp_dir)
    sys.argv = [
        'dummy.py',
        "--prompt",
        art_description,
        "--download-count",
        "1",
        "--output-dir",
        temp_dir,
        "-U",
        "1EjoFzkJCM7SIFi83ZBMI7MEqzTfta4Tn13GdB1Nl19336orPEXM6vzuaM79K4i1WofxCBLsypVtQA072F1aHiG9Oit_c4aYOL_sPNARNBLCwPD1JTRFQgLWtdwhZ4KBf6Jrq5J1D3Dvs3tokwLKy5LfJ9Uwh_HzZ2pSJrjPGG2Av2HLnIZrVKzlR3LZqnU2ypWfnxamreh_Qlfrx-aCDzg"
    ]
    
    BingImageCreator.main()
    # # Find the jpeg file in the temp_dir
    jpeg_file = next(pathlib.Path(temp_dir).glob('*.jpeg'))
    # jpeg_file = pathlib.Path('/var/folders/vb/j4ndg33n0wx40znrr0rr4p2h0000gn/T/tmpszp4zfg5/0.jpeg')

    # # Truncate the top ninth and bottom two ninths of the image
    # img = Image.open(jpeg_file)
    # width, height = img.size
    # new_height = int(height * 6 / 9)  # Keep the middle six-ninths of the image
    # top = int(height * 1 / 9)  # Start cropping from one-ninth of the image height
    # print('here', (0, top, width, top + new_height))
    
    # img_cropped = img.crop((0, top, width, top + new_height))
    # # img_cropped.save(str(jpeg_file).split('.')[0] + 'cropped.jpeg')
    # cropped_path = str(jpeg_file).split('.')[0] + 'cropped.jpeg'
    # img_cropped.save(cropped_path)

    # Upload the jpeg file to cloudinary
    # cloudinaryUploadResult = cloudinary.uploader.upload(cropped_path)
    corrected_aspect_ratio_url = correct_aspect_ratio(jpeg_file, art_description)

    print('just to be clear, this is the art url we are giving to mtg.design', corrected_aspect_ratio_url)

    sys.argv = original
    return corrected_aspect_ratio_url

def get_stablediffusion_url(art_description):


    # output = replicate.run(
    #     "cjwbw/stable-diffusion-v2:e5e1fd333a08c8035974a01dd42f799f1cca4625aec374643d716d9ae40cf2e4",
    #     input={
    #         "prompt": art_description, 
    #         "width":512, 
    #         "height":384}
    # )

    output = replicate.run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={
            "prompt": art_description,
            "width":512,
            "height":384
        }
    )
    # model = replicate.models.get('stability-ai/stable-diffusion')
    # results = model.predict(
	# 		prompt=art_description,
	# 		num_inference_steps=50,
	# 		num_outputs=1,
    #         width
	# 	)
    
    cloudinaryUploadResult = cloudinary.uploader.upload(output[0], )
                        # public_id=f'{roomCode}/{promptId}/{index}')
    return cloudinaryUploadResult['url']

def get_dalle_url(art_description):
        # Define the parameters for the image creation API
    image_params = {
        "prompt": art_description,
        # "temperature": 0.5,
        "size": "256x256",
    }

    # Call the OpenAI Image creation API
    image_response = openai.Image.create(**image_params)
    print('img', image_response)
    # Get the image URL from the response
    image_url = image_response["data"][0]["url"]

    return image_url
    

# https://janekb04.github.io/py2gpt/
schema = \
[
    {
        'name': 'create_card_image',
        'description': 'Creates the image for a Magic: The Gathering card with the provided details.',
        'parameters': {
            'type': 'object',
            'properties': {
                'card_name': {
                    'description': 'The name of the card.',
                    'type': 'string'
                },
                'mana_cost': {
                    'description': 'The mana cost of the card.', # Keep in mind that the cost should be commensurate with the effect',
                    'type': 'string'
                },
                'rules_text': {
                    'description': 'Describes any triggered, activated, static abilities the card has if applicable.',
                    'type': 'string'
                },
                'card_type': {
                    'description': 'The type of the card (e.g., Artifact, Creature, Land, Instant, Sorcery, Enchantment, Planeswalker).',
                    'type': 'string',
                    'enum': (
                        'Artifact',
                        'Creature',
                        'Land',
                        'Instant',
                        'Sorcery',
                        'Enchantment',
                        'Planeswalker'
                    )
                },
                'flavor_text': {
                    'description': 'The flavor text of the card. This should be omitted when the rules text is long',
                    'type': 'string'
                },
                'rarity': {
                    'description': 'The rarity of the card (e.g., Common, Uncommon, Rare, Mythic Rare).',
                    'type': 'string',
                    'enum': ('Common', 'Uncommon', 'Rare', 'Mythic Rare')
                },
                'power': {
                    'description': 'The power of the card (relevant for Creature type cards).',
                    'type': 'integer'
                },
                'toughness': {
                    'description': 'The toughness of the card (relevant for Creature type cards).',
                    'type': 'integer'
                },
                'art_description': {
                    'description': 'The description of the art on the card.',
                    'type': 'string'
                },
                'explanation': {
                    'description': 'Explanation for how the card satisfies the prompt.',
                    'type': 'string'
                }
            }
        },
        'required': [
            'card_name',
            'mana_cost',
            'rules_text',
            'card_type',
            'flavor_text',
            'rarity',
            'power',
            'toughness',
            'art_description',
            'explanation'
        ]
    }
]



from mtg_card_table import MTGCard
def get_card_outline(prompt, original: Optional[MTGCard], mode: str) -> dict:
    is_copy = mode == 'copy'
    if is_copy:
        print('Calling COPY version of gpt')
        messages = [
            {"role": "system", "content": "You design Magic The Gathering cards interactively." +
                " The user wants to base their card off of this one. In addition to whatever" +
                " changes the user describes, change the name and art"},
            # {"role": "user", "content": f"Create a Magic The Gathering card like {original.prompt}"},
            {
                "role": "function",
                "name": "create_card_image",
                "content": json.dumps(original.card_details),
            },
            {"role": "user", "content": prompt}
        ]
    elif original:
        print('Calling EDIT version of gpt')
        messages = [
            {"role": "system", "content": "You design Magic The Gathering cards interactively." +
                " If the user has feedback, don't change anything unless they ask."},
            {"role": "user", "content": f"Create a Magic The Gathering card like {original.prompt}"},
            {
                "role": "function",
                "name": "create_card_image",
                "content": json.dumps(original.card_details),
            },
            {"role": "user", "content": prompt}
        ]
    else:
        print('Calling CREATE version of gpt')
        messages = [{"role": "user", "content": f"Create a Magic The Gathering card like {prompt}"}]

    functions = schema
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        functions=functions,
        # function_call="auto",  # auto is default, but we'll be explicit
        function_call={"name": "create_card_image"}
    )
    response_message = response["choices"][0]["message"]
    print('made a card?', response_message)
    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        # available_functions = {
        #     "create_card_image": create_card_image,
        # }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        # fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        # function_response = fuction_to_call(
        #     location=function_args.get("location"),
        #     unit=function_args.get("unit"),
        # )

        # # Step 4: send the info on the function call and function response to GPT
        # messages.append(response_message)  # extend conversation with assistant's reply
        # messages.append(
        #     {
        #         "role": "function",
        #         "name": function_name,
        #         "content": "Stubbed out",
        #     }
        # )  # extend conversation with function response
        # second_response = openai.ChatCompletion.create(
        #     model="gpt-4-0613",
        #     messages=messages,
        # )  # get a new response from GPT where it can see the function response
        # if 'explanation' not in function_args:
        #     function_args['explanation'] = 'Missing'
        # if 'power' not in function_args:
        #     function_args['power'] = 0
        # if 'toughness' not in function_args:
        #     function_args['toughness'] = 0
            
        # return MTGCard(**function_args)
        return function_args
    else:
        raise Exception('should not have done this. shoulda called the func!')

# def generate_card_jpg(card_data: str):

#     pass

from mtg_card_table import MTGCard
from pprint import pprint
from typing import Iterator, Tuple
import concurrent.futures
import time
import uuid

def generate_card_final(description: str, original_card_id: str=None, author_user_id: str=None, mode:str=None) -> Iterator[Tuple[MTGCard, str]]:
    # card: MTGCard = MTGCard(prompt = description)
    # card.save()
    # todo record parent
    card = MTGCard.init_new_row(description, original_card_id, author_user_id)
    original = MTGCard.get(original_card_id) if original_card_id else None
    yield card, 'Skeleton'
    # raw = get_card_data(description)
    # structured = MTGCard(**raw)
    outline: dict
    if os.environ.get('IS_DEBUG') and False:
        # raise Exception('oops')
        outline = {
            "card_name": str(uuid.uuid4()),
            "rules_text": '0: do nothing\n1: exile target creature',
            "mana_cost": "1",
            "card_type": "Planeswalker",
            "flavor_text": "This is a placeholder flavor text.",
            "rarity": "Common",
            "power": 1,
            "toughness": 1,
            "art_description": "This is a placeholder art description.",
            "explanation": "This is a placeholder explanation."
        }
    else:
        outline = get_card_outline(description, original, mode)
    # card = MTGCard(
    #     **outline,
    #     prompt=card.prompt,
    #     id=card.id
    # )
    for key, value in outline.items():
        setattr(card, key, value)
    card.save()
    yield card, 'Outlined'

    # print('card outline')
    # pprint(card.attribute_values)
    # card.prompt = description

    temp_dir = tempfile.mkdtemp()

    art_description = (card.art_description or '').rstrip('.') + '. In the style of high quality epic fantasy digital art'
    original_art_description = original.art_description if original else ''
    if original_art_description == card.art_description:
        card.art_url = original.art_url
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if os.environ.get('IS_DEBUG'):
                global get_art_url
                get_art_url = lambda *args: 'https://th.bing.com/th/id/OIG.9.jDik1fO.pL1oJcw2c7?pid=ImgGn'
            future = executor.submit(get_art_url, art_description, temp_dir)
            while True:
                time.sleep(3)
                if future.done():
                    art_url = future.result()
                    card.art_url = art_url
                    card.save()
                    yield card, 'Artwork'
                    break
                else:
                    yield card, 'KeepAlive'

    # art_url = get_art_url(
    #     card.art_description.rstrip('.') + '. In the style of high quality epic fantasy digital art', 
    #     temp_dir)
    # card.art_url = art_url
    # card.save()
    # yield card, 'Artwork'

    final_render = render_mtg_card(temp_dir, card)
    cloudinaryUploadResult = cloudinary.uploader.upload(final_render)
    print('uploaded final render to', cloudinaryUploadResult['url'])
    card.final_rendered_url = cloudinaryUploadResult['url']
    card.is_finished_generating = True
    card.save()
    if original and mode == 'edit':
        original.is_superseded = True
        original.save()
    yield card, 'Rendered'
    # return card
    # print('creating')
    # pprint(raw)
    # final = create_card_image(**raw)
    # return {
    #     **raw,
    #     'rendered_card': final 
    # }

if __name__ == '__main__':
    # print(get_bing_url("High quality epic fantasy oil painting of a mage casting a spell of nullification"))
    # exit()
    # url=get_stablediffusion_url('high fantasy oil painting of eerie pirates')
    # print(url)
    # 1/0
    # from pprint import pprint
    raw=(get_card_data('A massive green creature that is totally broken'))
    # pprint(raw)
    
    final = create_card_image(**raw)
    # final = create_card_image(**{
    #     'art_description': 'A shadowy figure is shown, ready to stab a green elfin '
    #                     'creature kneeling in supplication.',
    # 'card_name': 'Blah Sacrificer',
    # 'card_type': 'Creature',
    # 'flavor_text': '"Your services are no longer required."',
    # 'mana_cost': '0',
    # "power": 10,
    # 'rarity': 'Rare',
    # 'rules_text': 'Tap, Sacrifice a creature that taps for mana: Add two mana of '
    #             'any one color.'
    #             }
    # )
    print('this is final', final)
