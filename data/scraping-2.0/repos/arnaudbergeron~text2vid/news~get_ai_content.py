import openai
import os
import BingImageCreator
import configparser
from PIL import Image


def get_prompt_from_script(script, chosen_headlines=None):
    model = "gpt-3.5-turbo"
    headlines = [i['title'] for i in script]
    index_chosen = []
    if chosen_headlines is None:
        prompt_headline = f"""{HEADLINE_CHOOSE_PROMPT} {headlines}. NO PROSE"""
        headlines_chosen = get_gpt_response(prompt=prompt_headline, model=model)
        index_chosen_str = headlines_chosen.replace("[", "").replace("]", "").split(",")
        for i in index_chosen_str:
            #keep only the numbers
            i = ''.join(filter(str.isdigit, i))
            i = int(i)
            index_chosen.append(i)

    else:
        index_chosen = chosen_headlines

    print('Chosen headlines:')
    for i in index_chosen:
        print(script[i]['title'])

    prompt_script = []
    for idx, i in enumerate(index_chosen):
        info = script[i]['summary']
        if idx == 0:
            pre_text = 'first'
        elif idx == 1:
            pre_text = 'second'
        elif idx == 2:
            pre_text = 'third'
        else:
            pre_text = ''
        prompt_script.append(f'''{SCRIPT_PROMPT} Here is your {pre_text} article you will report on: {info} NO PROSE''')


    to_tts = []
    print('Getting the script from gpt')
    for prompt in prompt_script:
        response = get_gpt_response(prompt=prompt, model=model)
        to_tts.append(response)
    return to_tts

def get_pictures_from_script(gpt_script, id):
    config = configparser.ConfigParser()
    config.read('config.ini')
    bing = BingImageCreator.ImageGen(config["Video"]["bing"]) #https://github.com/acheong08/BingImageCreator
    image_path = []
    print('Getting the images from bing')

    for num, sub_script in enumerate(gpt_script):
        for rem in ['<p>', '</p>', '<s>', '</s>', '<speak>', '</speak>']:
            true_script = [ss.replace(rem, '') for ss in sub_script]

        script_to_gen = f""""{IMAGE_PROMPT} {true_script} NO PROSE"""
        prompt_img = get_gpt_response(prompt=script_to_gen, model="gpt-3.5-turbo")
        img = bing.get_images(prompt_img)
        bing.save_images(img, f'/Users/arnaudbergeron/Desktop/Code/auto_content/assets/bingOutput', file_name=f'{id}_{num}')
        image_path.append([f'/Users/arnaudbergeron/Desktop/Code/auto_content/assets/bingOutput/{id}_{num}_{i}.jpeg' for i in range(0, len(img))])

    for i in image_path:
        for j in i:
        #crop the image to 578x1024 using pillow
            img = Image.open(j)
            #crop image to 578x1024 from center
            img = img.crop((27, 0, (27+984), 984))
            img.save(j)
    return image_path

def get_gpt_response(prompt, model):
    config = configparser.ConfigParser()
    config.read('config.ini')
    openai.api_key = config["Video"]["open_ai_secret"]

    prompt = prompt.encode('ascii', 'ignore').decode('ascii')

    message = [{"role": "user",'content':prompt}]
    response = openai.ChatCompletion.create(
                    model=model,
                    messages=message,
                    temperature=0.5,
                    frequency_penalty=0.0,
        presence_penalty=0.0
        )['choices'][0]['message']['content']
    return response


HEADLINE_CHOOSE_PROMPT = """You are now an expert news editor for short form content. \ 
I will give you headlines and information and I need you to select 3 of the best headlines \ 
for a young business minded audience that uses tiktok in a bullet information style for a 1 minute \
video to inform people of the news of the day.  \
Your first task is to choose the 5 best headlines from the following list. \
Make sure to output only the index of the headlines you want to report on in \
order in which you think your target audience will enjoy more, \
an example output would be: \
"[headline_index, headline_index, headline_index, headline_index, headline_index]" \
where the first headline has index 0. \
Choose the best headlines about companies and issues that most millenials are interested \
about and will care to hear about. You need to choose the headlines you think will resonate \
the most with your audiance choose the biggest headlines, you only have 60 seconds of their time! \
Here are the healines: """

SCRIPT_PROMPT = '''You are now an expert news anchor for short form content \
for a young business minded audience that uses tiktok in a bullet information \
style for a 1 minute video to inform people of the news of the day. \ 
Your task is to write a script for a 8 seconds segment on the following article and \ 
only output the script no prose. You need to be accurate with the facts, interesting and succinct. \
Be exciting you want the audiance to be captivated in the subject you are reporting on. \
Make sure to focus more on the story rather than the numbers. \
Use as little abbreviations as possible, if you have to use some give me their phonetic representation:
for example AI would be replaced by Artificial Intelligence.\
Output at max 2 or 3 sentences.'''

IMAGE_PROMPT = """"You are an expert creative director for a for short form news content platform. \
I will give you the script for the show and I need you to generate a prompt that will be used \
to create the images for the videos. Be sure to be very specific about location, emotions, \
composition and perspective, you want the prompt to generate a very realistic picture include \
a high amount of details as if the image was real. You do not want to be cliche for the image nor \
include any faces if you need to potray people you should show them looking away from the camera. \
Here's an example instead of the prompt: An image of the US Capitol building with the words \
'debt ceiling' in bold letters., you should say:  Create an image showcasing the iconic US \
Capitol building, symbolizing the concept of the debt ceiling. Capture the building from a striking \
perspective, emphasizing its grandeur and architectural details. Depict the Capitol building \
under a clear sky, with the sunlight casting a warm glow. Frame the image to highlight the \
building's dome, columns, and other distinctive features. Avoid including any faces or human \
figures in the image. Additionally, please refrain from incorporating any letters or specific words, \
ensuring the focus remains solely on the Capitol building and its symbolic representation of the debt \
ceiling. Here's the prompt for today's show:"""