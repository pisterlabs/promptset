import openai
import json
import random
import os
import uuid 
import sys
import requests
from tempfile import TemporaryDirectory
from google.cloud import storage
from . import wikipedia
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def story(openai_api_key, categories=None, google_id=None, google_bucket=None):
    """
    Generates a story based on categories, parses it into sections, generates images for each section, and converts the story and images into HTML.
    """
    openai.api_key = openai_api_key
    categories = cleaned_categories(categories)
    story = generate_story(categories)
    try:
        title, sections = parse_story(story)
    except ValueError as e:
        logging.info(f"Error parsing following story: {story}")
    image_urls = image_generator(sections)
    if google_id is not None and google_bucket is not None:
        image_urls = upload_files_to_bucket(image_urls, google_id, google_bucket) 
    html = html_converter(title, sections, image_urls, categories)
    return html

def upload_files_to_bucket(image_urls, google_id, google_bucket):
    """Downloads images from URLs, saves them in a temporary directory, and uploads them to Google Cloud Storage."""
    client = storage.Client(project=google_id)
    bucket = client.get_bucket(google_bucket)
    uploaded_image_urls = []
    headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}
    with TemporaryDirectory() as temp_dir:
        for image_url in image_urls:
            response = requests.get(image_url, headers=headers)
            response.raise_for_status()
            image_extension = os.path.splitext(image_url)[1]
            unique_image_name = str(uuid.uuid4()) + image_extension
            image_path = os.path.join(temp_dir, unique_image_name)
            with open(image_path, 'wb') as f:
                f.write(response.content)
            blob = bucket.blob(unique_image_name)
            blob.upload_from_filename(image_path)
            uploaded_image_url = blob.public_url
            uploaded_image_urls.append(uploaded_image_url)
    return uploaded_image_urls


def cleaned_categories(categories):
    """
    Returns a string of categories, either from the provided list or by generating random categories.
    """
    if categories is None:
        categories = random_categories()
    else:
        categories = ", ".join(categories)
    return categories


def random_categories():
    """
    Reads the data.txt file and returns a string of three random categories.
    """
    # Get the directory of this script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Get the path of data.txt
    data_path = os.path.join(dir_path, "data.txt")
    with open(data_path, "r") as f:
        lines = f.readlines()
    categories = random.sample(lines, 3)
    categories = [category.rstrip("\n") for category in categories]
    categories_str = ", ".join(categories)
    return categories_str


def parse_story(story_str):
    """
    Parses a JSON-formatted story string into a title and a list of sections.
    """
    story_dict = json.loads(story_str)
    title = story_dict.get("title")
    year = story_dict.get("year")
    sections = [
        value for key, value in story_dict.items() if key not in ["title", "year"]
    ]
    return title, sections


def gpt(temperature, messages, model):
    """
    Sends a chat completion request to the OpenAI API and returns the content of the response.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content


def dalle(prompt, model="dall-e-3", size="1024x1024", quality="hd", n=1):
    """
    Sends an image creation request to the OpenAI API and returns the URL of the generated image.
    """
    response = openai.Image.create(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )
    image_url = response.data[0].url
    return image_url


def prompts(categories, random_year):
    """
    Returns three prompts for generating a story, based on the provided categories and a random year.
    """
    first_prompt = f"""\
    I want you to be my cowriter on innovative and ambitious future ideas which are still in the realm of physics so that its not unrealistic that something like this exists. Following and example of such and scenario:

    ---

    Introducing "Skill Factory" 
    Beginning 2026

    College has failed America. Shop classes are in decline. We are indebted to institutions that no longer serve us. 

    Millions of opportunities grow stale as America's future decays

    It's Time To Rebuild 

    ...

    Welcome to Skill Factory #001

    Designed with your local Employer/Sponsor rather than state-funded

    The simplest way to train for & pursue today's careers

    There are no entry exams, schedules, or pre-requisites.

    No Professors. No Tuition. 

    Download the app and let’s explore

    ...

    Screens across the walls list local 'Opportunities', from the Sponsors

    - Lockheed Layup Tech - $23/hour
    - Northrop CMM Tech - $38/hour
    - Tesla Robotics - $52/hour

    Tesla Robotics? Great choice. Only 15 miles away. 

    Let's scan the code to check it out

    ...

    During setup of your app, the algorithm baselines your "Skill Profile" from your background

    Let's watch it compare your Skill Profile and this Tesla Robotics position

    68% Ready, Great. Only 14 skills missing. This should only take a few weeks.

    Let's head over to Training

    ...

    As we walk over, you'll notice each area is dedicated to a set of selected technologies and skills

    You can reserve space where you'd like to explore. Some rooms may require a safety prep for activation

    These are here 24/7/365 so you can choose times that work for you

    ...

    Training:

    A Skill Path has been generated for you, guiding you through the remaining 14 skills

    You'll build projects specifically approved by Tesla's hiring & engineering team

    Use the AI, AR, & Training modules to aid you

    Upload your project and earn the Tesla Certification

    ...

    Overwhelmed?

    Watch for days where professionals from Sponsors will be spending time here

    They may be exploring new technologies or training new skills for their own skill profile

    They are here to support you. The highest reviewed Mentors are highlighted in the Hall of Fame

    ...

    As you progress you'll see more opportunities in your app light up. 

    You qualify to interview for these instantly

    In your "Careers" tab, you'll find Hiring Managers that have offers or are requesting for you to apply

    Congratulations and good luck on your new journey 🇺🇸❤️

    ...

    Why are we building these?

    Companies waste billions every year to search for people that don't exist

    Meanwhile individuals face slow & expensive options with little guidance

    With capacity to help 100,000+ annually per Skill Factory

    We see a way to unlock a Century of Progress

    ...

    If your company would benefit from an endless pipeline of precisely trained candidates

    Or if you wish to have your hardware / software available in America's greatest "Industry Showroom"

    Drop a note below and follow along as we bring this to the world

    It's Time We Rebuild

    --- 

    The example was about an innovative school concept. There are no limits for which areas you can create the concept for. I will provide you with a few categories which should be involved in the concept and you should create the concept from there. please focus on the science aspects and that its realistic and not to much science fiction. it also shouldnt be just the story about it but rather a excursion for the reader like if he would be there. The story should play in the year {random_year}, i.e. build the scenario realistically and not to far fetched - we are currently in the year 2023.
    """

    second_prompt = f"""\
    Absolutely, I'd be delighted to collaborate with you on creating innovative concepts grounded in the realm of physics and the real world. To begin, you can provide me with a few categories or areas of focus, and I will craft a concept around them, ensuring that the ideas are scientifically plausible, innovative and that its like the view of an person who is there.

    For instance, if you're interested in areas like renewable energy, space exploration, advanced robotics, healthcare technology, or smart cities, just let me know. I will then develop a concept that integrates these categories, keeping in mind the principles of physics and the feasibility of such technologies in the near future.

    Feel free to share the categories or specific areas you're interested in, and we can start brainstorming and shaping these futuristic concepts together! Evertything will be realistic.
    """

    third_prompt = f"""\
    the categories are: {categories}. dont focus on to many different concepts but rather narrow it down a bit. please go ahead and write the short story based on this categories. only the JSON of the story should be returned in JSON - this is really important - as following:

    ---
    {{
        "title": "title"
        "first_section": "first section"
        "second_section": "second section"
        "third_section": "third section"
        "fourth_section": "fourth section"
        "fifth_section": "fifth section"
        "sixth_section": "sixth section"
        "seventh_section": "seventh section"
        "eighth_section": "eighth section"
        "ninth_section": "ninth section"
        "tenth_section": "tenth section"
    }}
    --- 

    JSON:
    """
    return first_prompt, second_prompt, third_prompt


def generate_story(categories):
    """
    Generates a story based on the provided categories by sending prompts to the OpenAI API.
    """
    random_year = random.randint(2025, 2035)
    first_prompt, second_prompt, third_prompt = prompts(categories, random_year)
    messages = [
        {"role": "user", "content": first_prompt},
        {"role": "assistant", "content": second_prompt},
        {"role": "user", "content": third_prompt},
    ]
    # gpt-3.5-turbo-1106
    story = gpt(0.5, messages, model="gpt-4")
    return story


def generate_first_image(first_section):
    """
    Generates an image for the first section of the story by sending a prompt to the OpenAI API.
    """
    prompt = f"""\
    i generated a short story with ten short sections. my goal is it now to create an appropriate image for each section. the section i am going to provide to you is also the section which is responsible for the title image (the image which serves as thumbnail for the story). please generate an image for the following section:

    ---
    {first_section}
    ---

    please keep in mind that the image you generate shouldnt be to far fetched into the future and realistic in the realm of physics. IT IS IMPORTANT THAT THE IMAGE IS REALISTIC!!!
    """
    image_url = dalle(prompt)
    return image_url


def generate_image(section):
    """
    Generates an image for a given section of the story by sending a prompt to the OpenAI API.
    """
    prompt = f"""\
    i generated a short story with ten short sections. my goal is it now to create an appropriate image for each section. please generate an image for the following section:

    ---
    {section}
    ---

    please keep in mind that the image you generate shouldnt be to far fetched into the future and realistic in the realm of physics. IT IS IMPORTANT THAT THE IMAGE IS REALISTIC!!!
    """
    image_url = dalle(prompt)
    return image_url


def image_generator(story):
    """
    Generates images for each section of the story, either by finding an image on Wikipedia or by generating an image using the OpenAI API.
    """
    image_urls = []
    random_sections = random.sample(story[1:], 5)
    for line in story:
        # if first line than call generate_first_image function
        if line == story[0]:
            image_url = generate_first_image(line)
            image_urls.append(image_url)
        # if line is in random_sections, call wikipedia.find_image_for_story_section
        elif line in random_sections:
            # returns image url or None
            image_url = wikipedia.find_image_for_story_section(line, openai.api_key)
            if image_url == "no images found":
                image_url = generate_image(line)
            image_urls.append(image_url)
        # otherwise call generate_image function
        else:
            image_url = generate_image(line)
            image_urls.append(image_url)
    return image_urls


def html_converter(title, sections, image_urls, categories):
    """
    Converts the title, sections, images, and categories of a story into an HTML string.
    """
    html = []
    html.append("<html>\n")
    html.append("<head>\n")
    html.append("<style>\n")
    html.append(".section, .categories { font-size: 15px; font-weight: normal; }\n")
    html.append("</style>\n")
    html.append("</head>\n")
    html.append("<body>\n")
    html.append(f"<h1><strong>{title}</strong></h1>\n")
    html.append(f'<h3 class="categories"><em>Categories: {categories}</em></h3>\n')
    for section, image_url in zip(sections, image_urls):
        html.append(f'<div class="section">\n')
        html.append(f"<p>{section}</p>\n")
        html.append(
            f'<img src="{image_url}" alt="Image for {section}" width="256" height="256">\n'
        )
        html.append("</div>\n")
    html.append("</body>\n")
    html.append("</html>\n")
    return "".join(html)
