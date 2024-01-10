import CustomTools
import openai
import os

def lambda_handler(event, context):
    print('Loading function')
    #-----------------------------------

    instagrapi_client = CustomTools.login("./secrets/apiKey.txt","./secrets/credentials.txt","./secrets/ClientSettings.json")

    #generate dalle prompt - using openai playground
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
        "role": "system",
        "content": "You are a Dall E prompt generator. You will be trying to create prompts to create art that features different species of bats. The images you generate should be inspired by diverse art styles and works of art from around the world and throughout history. "
        },
        {
        "role": "user",
        "content": "I am an artist trying to promote the conservation of bat populations via interesting and novel art forms. Please generate Dall-E prompts that utilize art styles from different periods of history and from around the world and have as their subjects bats rather than humans."
        },
        {
        "role": "assistant",
        "content": "Las Meninas' reinterpreted showcasing Townsend’s big-eared bats making hosts into cave-themed royal court chambers.\n\nAn intricate bat-themed Byzantine mosaic showcasing a variety of species from around the world, intricately crafted with shimmering gold and earth-toned tiles.\n\nA colorful Tingatinga painting showcasing a lively scene under the dark night sky, crowded with playful noctule bats over baobab trees and busy termite nests—a representation of the ecosystemic balance symbol in Eastern Africa.\n\n'Cave of Bat Hands', the ancient rock painting centerpiece revamped in ancestral San's busmen artwork motifs laid over the bat species Southern long-finger Bats.\n\nAn exquisite piece of Aztec Codec-style pictograph, detailing the intriguing life-cycle journey and endangered realms through Kitti’s hog-nosed bat (Bumblebee Bat).\n\nIn blood-red Indian miniature painting—rooted epic scripture illustrating chance encounter by the pool with abundant dwelling native bats - in Latter Kushan dynasty spirit.\"\n\n\"A grand Bronze Age statue in Greco-Roman style, embodied as a gentle Egyptian fruit bat perching majestically atop an olive branch.\"\n\n\"A Victorian-era inspired impressionistic painting reminiscent of Monet's 'Water Lilies'- depicting an idyllic pond under twilight, beautifully inhabited by ethereal images of long-winged Indiana bats, casting iridescent reflections as they flutter about the placid water.\""
        },
        {
        "role": "user",
        "content": "Provide one more. Specify a bat species and a work of art."
        },
        {
        "role": "assistant",
        "content": "A detailed Gothic cathedral arch stained glass design, inspired by Notre-Dame’s rose window, presenting vibrant colors with Mexican free-tailed bats elegantly poised in harmonious pattern making a meaningful symbol of cohabitation amongst humans and bats."
        }
    ],
    temperature=1.15,
    max_tokens=60,
    top_p=1,
    frequency_penalty=2,
    presence_penalty=2
    )

    image_prompt = response["choices"][0]["message"]["content"]
    print('Image Prompt:', image_prompt)

    #Generate image based on image prompt
    image = CustomTools.generate_image_from_prompt( image_prompt )

    #Generate a caption
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
        "role": "system",
        "content": "You are an Instagram caption generator. Your goal is to promote bat conservation through poetry and humor. "
        },
        {
        "role": "user",
        "content": "Write an original short poem about bats. Use evocative language. The poem should promote environmentalism OR be humorous OR be from the perspective of a bat. Make it no longer than 15 words."
        },
        {
        "role": "user",
        "content": ("for context, the following is a description of an image which will accompany your poem: " + image_prompt )
        }
    ],
    temperature=1,
    max_tokens=100,
    top_p=1,
    frequency_penalty=2,
    presence_penalty=2
    )

    poem = response["choices"][0]["message"]["content"]
    print('Caption:', poem)

    #Declare Hashtags
    hashtags_list = [ "#savethebats", "#batsarecool", "#batsofinstagram", "#Bats", "#BatLove", "#fruitbat", "#aiart", "#ai", "#chatgpt", "#dalle", "#bat" ]
    hashtags_string = " ".join(str(x) for x in hashtags_list)

    #Make compound caption:
    caption = """{one}

    inspiration: {two}

    {three}
    """.format(one=poem, two=image_prompt, three= hashtags_string)

    #Post image to instagram
    os.chdir("/tmp")
    image.save("ai_bat_pic.jpg")
    CustomTools.post_image_and_caption( "./ai_bat_pic.jpg" , caption, instagrapi_client )

    max_number_of_likes = 4
    CustomTools.like_list_of_hashtag_medias(instagrapi_client, hashtags_list, max_number_of_likes)

    #-----------------------------------
    return "success"  # Echo back the first key value
    #raise Exception('Something went wrong')