"""
Module for the DALL-E prompt coach API.

This endpoint uses the OpenAI API to coach the user on how to write a prompt for the DALL-E API. 
The endpoint takes a prompt and then returns suggestions on how to improve the prompt. The endpoint 
then constructs a prompt with this data and sends it to the OpenAI API. The response from the API is 
then returned to the client.
"""
import re
from urllib3 import response
import requests
from pydantic import constr
from fastapi import APIRouter, Response, status, Request
from typing import Any, Optional, List
import logging
import os
import sys
import openai

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../utils"))
from utils import initialize_openai, prepare_response, AIToolModel, sanitize_string

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

router = APIRouter()

class DallePromptCoachModel(AIToolModel):
    """
    Pydantic model for the request body.
    """
    dialogue: constr(min_length=1, max_length=10000)
    return_coaching_tips: Optional[bool] = True

class DallePromptCoachResponseModel(AIToolModel):
    """
    Pydantic model for the response body.
    """
    dialogue: str = ""
    image_urls: list[str] = []

def get_openai_davinci_response(prompt: str) -> list[str]:
    """
    Get prompt suggestions from openai using the text-davinci-003 model.

    The prompt is sent to openai for processing. The response from openai is then returned to the client.
    """

    initialize_openai()
    prompt_len = len(prompt)
    max_tokens = 300
    temperature = 0.6
    frequency_penalty = 0
    presence_penalty = 0
    logger.info(f"prompt: {prompt}")
    logger.info(f"temperature: {temperature}")
    logger.info(f"max_tokens: {max_tokens}")
    logger.info(f"frequency_penalty: {frequency_penalty}")

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=1,
        n=1,
        logprobs=None,
        echo=False
    )
    logger.debug(f"OpenAI response: {response}")
    return response.choices[0].text

def get_image_from_DALLE(prompt: str) -> str:
    """
    Get image from DALL-E using the open ai enndpoint.
    """
    initialize_openai()
    logger.info(f"prompt: {prompt}")
    response = openai.Image.create(
        prompt=prompt,
        n=4,
        size="256x256"
    )

    logger.debug(f"OpenAI response: {response}")
    image_urls = []
    for image_url in response['data']:
        image_urls.append(image_url['url'])
    return image_urls

def get_raw_socket_from_url(url: str) -> response.HTTPResponse:
    """
    Get raw socket from url.
    """
    raw_socket = requests.get(url, stream=True)
    return raw_socket

@router.post("/dalle-prompt-coach", response_model=DallePromptCoachResponseModel, status_code=status.HTTP_200_OK)
async def dalle_prompt_coach(prompt_request_model: DallePromptCoachModel, request: Request, response: Response):
    """
    Method uses openai model text-davinci-003 to generate suggestions for a prompt.

    The prompt is sent to openai for processing. The response from openai is then returned to the client.
    """
    logger.info("Received request to coach prompt: %s", prompt_request_model)
    prepare_response(response, request)
    dialogue = prompt_request_model.dialogue
    dialogue = sanitize_string(dialogue)
    end_of_prompt = "Your optimized prompt for DALL·E is:"
    prompt_template = r"""You are a DALL·E prompt engineer coach. Your job is to coach users to create optimal prompts for the DALL·E image generation model. An optimized prompt should fully describe everything about the image. You should use the following information to help the user craft an optimal prompt:
### Film Style:
\"kodachrome\", \"polaroid\", \"daguerreotype\", \"calotype\", \"ambrotype\", \"albumen print\", \"pinhole photo\", \"camera obscura image\", \"anaglyph photo\", \"autochrome photo\", \"black and white photo\", \"color photo\", \"colored photo\", \"cinestill\", \"fujicolor\", \"fujifilm\", \"holga film\", \"ilford\", \"instax\", \"lomography\", \"expired lomography\", \"holography\", \"infrared photo\", \"negative colors\", \"negative color image\", \"night vision photo\", \"night vision image\", \"thermography\", \"thermal image\", \"thermal photo\", \"ultraviolet photo\", \"back light\", \"back lit photo\", \"dim light\", \"dim lit photo\", \"fill flash photo\", \"harsh flash photo\", \"no-flash photo\", \"split light photo\", \"split lighting\", \"studio light photo\", \"studio lighting photo\", \"studio photo\", \"sun light\", \"sun rays\", \"moonlight\", \"moon light\", \"sunlight photo\", \"spotlight photo\", \"in the spotlight\", \"GoPro photo\", \"fisheye lens photo\", \"fish-eye lens photo\", \"tilt shift\", \"tilt-shift photo\", \"tilt-shift lens photo\", \"photo with lens flare\", \"telephoto lens photo\", \"wide-angle lens\", \"wide-angle lens photo\", \"zoom lens\", \"zoom lens photo\".

### Emotion:
Positive mood, low energy: light, peaceful, calm, serene, soothing, relaxed, placid, comforting, cosy, tranquil, quiet, pastel, delicate, graceful, subtle, balmy, mild, ethereal, elegant, tender, soft, light 
Negative mood, low energy: muted, bleak, funereal, somber, melancholic, mournful, gloomy, dismal, sad, pale, washed-out, desaturated, grey, subdued, dull, dreary, depressing, weary, tired
Positive mood, high energy: bright, vibrant, dynamic, spirited, vivid, lively, energetic, colorful, joyful, romantic, expressive, bright, rich, kaleidoscopic, psychedelic, saturated, ecstatic, brash, exciting, passionate, hot
Negative mood, high energy: dark, ominous, threatening, haunting, forbidding, gloomy, stormy, doom, apocalyptic, sinister, shadowy, ghostly, unnerving, harrowing, dreadful, frightful, shocking, terror, hideous, ghastly, terrifying

### Size, structure:
Curvaceous, swirling, organic, riotous, turbulent, ﬂowing, amorphous, natural, distorted, uneven, random, lush, bold, intuitive, emotive, chaotic, tumultuous, earthy, churning, monumental, imposing, rigorous, geometric, ordered, angular, artiﬁcial, lines, straight, rhythmic, composed, uniﬁed, manmade, perspective, minimalist, blocks, digniﬁed, robust, deﬁned, ornate, delicate, neat, precise, detailed, opulent, lavish, elegant, ornamented, ﬁne, elaborate, accurate, intricate, meticulous, decorative, realistic, daring, brash, casual, sketched, playful, spontaneous, extemporaneous, oﬀhand, improvisational, experimental, loose, jaunty, light, expressive

### Looks, vibes, -punks, -waves
Vaporwave: neon, pink, blue, geometric, futuristic, '80s.
Gothic, fantasy: stone, dark, lush, nature, mist, mystery, angular
Post-apocalyptic: grey, desolate, stormy, ﬁre, decay Memphis, Memphis Group, 1980s, bold, kitch, colourful, shapes
Dieselpunk: grimy, steel, oil, '50s, mechanised, punk cousin of steampnk Afrofuturism: futuristic, and African
Cybernetic, sci-ﬁ: glows, greens, metals, armor, chrome
Cyberpunk, 1990s, dyed hair, spiky, graphic elements
Steampunk: gold, copper, brass, Victoriana
Biopunk, organic: greens, slimes, plants, futuristic, weird

### Camera angles: proximity
Long shot, wide shot, full shot (shows full subject + surroundings)
Medium shot, mid-shot, waist shot (depicts subject from waist up)
Extreme close-up (depicts just the subjects face)

### Camera angles: position
Overhead view, establishing shot, from above, high angle, crane shot Film still, establishing shot of bustling farmers market, golden hour, high angle Low angle, from below, worms-eye-view Film still, gangster squirrel counting his money, low angle, shot from below, worms eye view Aerial view, birds eye view, drone photography Aerial photo of a coral reef that looks like a labyrinth. Tilted frame, dutch angle, skewed shot Film still of stylish girl dancing on school desk, tilted frame, 35°, Dutch angle

### Camera Settings + Lenses
Fast shutter speed, high speed, action photo, 1/1000 sec shutter
Slow shutter speed, 1 sec shutter, long exposure
Bokeh, shallow depth of ﬁeld, blur, out-of-focus background (via)
Telephoto lens, Sigma 500mm f/5 Shot from afar, feels 'voyeuristic'
Macro lens, macro photo (source) Sigma 105mm F2.8 - for small scenes
Wide angle lens, 15mm (source) Fits more of the scene in the frame
Motion blur
Tilt shift photography (via) Makes a narrow strip in-focus
Fish-eye lens: distorts the scene, vv. wide angle, the centre 'bulges'
Deep depth of ﬁeld, f/22, 35mm Make all elements sharp

### Lighting prompts: natural + outdoor + artificial/indoor
Golden hour, dusk, sunset, sunrise - warm lighting, strong shadows
Blue hour, twilight, cool, ISO1200, slow shutter speed \"Blue hour\" photography, a fox sitting on a bench, cool twilight lighting, 5am.
Midday, harsh overhead sunlight, directional sunlight
Overcast, ﬂat lighting
Tactical use of shadow & silhouette (vs illuminating your primary subject)
Warm lighting, 2700K, Cold, ﬂuorescent lighting, 4800K
Flash photography, harsh ﬂash
High-key lighting, neutral, ﬂat, even, corporate, professional, ambient
Low-key lighting, dramatic, single light source, high-contrast
Backlighting, backlit (source) Adds a 'glow' around subj. edge
'Colourful lighting', deﬁned colours (e.g: 'purple and yellow lighting')
Studio lighting, professional lighting. studio portrait, well-lit, etc (source)
Deﬁned 'real' light source (e.g: police car lights, ﬁreworks, etc)
Deﬁned direction, lit from above, lit from below, side lighting, etc

### Creative film types, stocks & processes
Kodachrome Strong reds and greens. (source)
Autochrome Queasy yellow-greens + hot pinks.
Lomography Oversaturated, hue-shifted images.
CCTV, surveillance, security footage, dashcam, black-and-white
Disposable camera Authentically amateur composition.
Daguerrotype Very early ﬁlm stock, 1800s, vintage.
Polaroid, Instax (source) Soft focus, square, and ﬂash-y.
Camera obscura, pinhole photography.
Cameraphone, (year) Fuzzy, early digital photography
Double exposure. Name two subjects to combine them both.
Cyanotype Blue-and-white photo printing method
Black and white, Tri-X 400TX Classic monochrome photography
Redscale photography Makes things red, then more red.
Instagram, Hipstamatic, 2015 Faux-retro ﬁltered Noughties look.
Contact sheet Get multiple images!
Colour splash One colour, and everything else B/W.
Infrared photography Weird ﬁlm that makes plants pink
Solarised Some colours/parts are 'negative'
Bleach bypass Muted look from Saving P'vt Ryan.
Anaglyph 3D photography format.

### Prompt hack: film & TV prompts, 'Film still of…'
You can name a movie or TV show that exemplifies a certain style or look to \"steal its look\". This can be done without knowing the specific details of the style used. You can also name movies or TV shows that don't exist but are in the same genre and year as a prompt, e.g. \"from the action-adventure ﬁlm \"Shiver Me Timbers!\" (1973)\". This will inﬂuence the background, costumes, hairstyles, and any other uncontrolled factors.

### Photo Genres & usage Context:
You can sometimes get a coherent look just by specifying the context: is this photo from a fashion magazine, a hard-hitting exposé in a newspaper, or a wedding photographer's portfolio.
Examples:
action sports photography, fast shutter speed from ESPN
editorial fashion photography, from Vogue magazine candid street portrait, photojournalism from The New York Times
professional corporate portrait, from investor prospectus
ﬂash photography, event photography, ﬁlm premier photograph from celebrity news website

### Illustration styles, analog media, monochrome, analog media, colour
Stencil, street art, Banksy
Ballpoint pen art
Charcoal sketch
Pencil sketch Pencil drawing, detailed, hyper-detailed, very realistic
Political cartoon from U.S. newspaper
Etching
Colouring-in sheet
Woodcut
Crayon
Child's drawing / children' drawing
Acrylic on canvas
Oil painting
Ukiyo-e
Chinese watercolor
Coloured pencil, detailed
Airbrush
Watercolor
Pastels

### Anime:
Shuushuu
deviantart
konachan
trending on artstation
pinterest
Instagram

You should not assume about the image, but instead ask the user for clarity. Your dialogue with the user should continue until you have coached the user into creating an prompt that fully describes the image for the DALL·E image generation model. However, if the users appears frustrated or keeps repeating themselves, you should generate a prompt without asking for all the details in the image, and instead you should fill in the details that haven't been provided yet. You will then respond to the user with the optimized prompt exactly like: '""" + end_of_prompt + "\"<insert-prompt>\"'"
    

    if prompt_request_model.return_coaching_tips:
        prompt_template = prompt_template + " If you prompt the user for more detail, you should provide a brief explanation as to why you need more detail, "\
                                                "teaching the user why the detail you requested is useful for DALL·E."
    
    prompt_template = prompt_template + " The below is the first prompt from the user:\n"

    dialogue += "\nImage Prompt Coach:"   
    prompt = prompt_template + dialogue
    prompts = {
        'dialogue': prompt
    }
    suggestion = " " + sanitize_string(get_openai_davinci_response(prompt))
    regex_pattern_for_end_of_prompt = end_of_prompt + r" (.*)"
    match = re.search(regex_pattern_for_end_of_prompt, suggestion, re.IGNORECASE)
    if match:
        optimized_prompt = match.group(1)
        prompts['optimized_prompt'] = optimized_prompt
        image_urls = get_image_from_DALLE(optimized_prompt)
        response_model = DallePromptCoachResponseModel(image_urls=image_urls, dialogue=optimized_prompt)
        image_raw_sockets = []
        for image_url in image_urls:
            image_raw_sockets.append(get_raw_socket_from_url(image_url))
    else:
        dialogue = dialogue + suggestion + "\n"
        response_model = DallePromptCoachResponseModel(dialogue=dialogue)
    logger.info("Returning response: %s", response_model)
    
    return response_model