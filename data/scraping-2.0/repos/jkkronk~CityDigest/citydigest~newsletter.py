from pydantic import BaseModel, Field
from openai import OpenAI
import instructor
from typing import Optional
import urllib.request

from pdf_content import pdf_to_string, get_all_pdfs_in_directory

class Article(BaseModel):
    title: str = Field(..., description="The title of the article.")
    text: str = Field(..., description="The 10 sentences long article that summarizes the best of the real newpaper article.")
    reference: str = Field(..., description="The reference of the article.")
    image_prompt: Optional[str] = Field(..., description="An optional image prompt of the article.")

class Newletter(BaseModel):
    title: str = Field(..., description="The title of todays newsletter.")
    introduction: str = Field(..., description="A digestible easygoing introduction of todays newsletter.")
    articles: list[Article] = Field(..., description="The articles of todays newsletter.")
    funny_joke: str = Field(..., description="A funny joke about city to end todays newsletter.")

def newsletter_mdformat(newsletter: Newletter, openai_api_key, generate_images=True):
    text = ""
    text += f"# {newsletter.title}\n"
    text += f"__{newsletter.introduction}__\n"
    for article in newsletter.articles:
        if article.image_prompt and generate_images:
            text += f"\n ![image]({generate_image(article.image_prompt, openai_api_key)}) \n"
        text += f"## {article.title} \n"
        text += f"{article.text} \n \n "
        text += f"*REFERENCE: {article.reference}*\n"

    text += f"## PS.\n"
    text += f"**{newsletter.funny_joke}**\n"

    text += f"## PPS.\n"

    text += f"__Thanks for reading and stay classy Zürich!__\n"
    return text

def get_newsletter(pdfs_directory, openai_api_key, country="Switzerland", city="Zurich", level="don't know much") -> Newletter:
    pdf_paths = get_all_pdfs_in_directory(pdfs_directory)

    previous_newsletter_iteration = "No previous newsletter."

    for pdf_path in pdf_paths:
        pdf_text = pdf_to_string(pdf_path)

        # Devide pdf_text into strings of 400000 characters and iterate over them
        # to avoid the 400000 character limit of OpenAI
        for i in range(0, len(pdf_text), 300000):
            input_string =  pdf_text[i:i + 300000]

            client = instructor.patch(OpenAI(api_key=openai_api_key))
            prompt = f"""
                    You are a news reporter that has been asked to summarize newspapers into a short newsletter.
                    The newsletter should only consist of news from {country} and {city}. 
                    The newsletter should be written for someone {level} about {country} and {city}.
                    The newsletter should be have about 5 articles. And each article should maximum be about 10 sentences long.
                    The newsletter with very personal and easygoing language. The newletter is called "Echo Echo {city}".
                    The newsletter should be written in English.
                    Provide references to the articles given the input. 
                    Add a prompt for an image that would be shown for some of the articles if you find it fitting.
                    Additionally, add a funny joke about {city} to end the newsletter.
                    
                    The current newspaper is as below. Use this and update new articles you seem to fit.
                    {previous_newsletter_iteration}
                    
                    Here is the text you should summarize:
                    {input_string}
                    """
            overview: Newletter = client.chat.completions.create(
                model="gpt-4-1106-preview",
                response_model=Newletter,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_retries=2,
            )

            print(f"Overview: {overview}")

            previous_newsletter_iteration = newsletter_mdformat(overview, openai_api_key, generate_images=False)

    return overview

def generate_image(promt: str, openai_api_key: str):
    client = instructor.patch(OpenAI(api_key=openai_api_key))
    response = client.images.generate(
        model="dall-e-3",
        prompt=promt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    urllib.request.urlretrieve(response.data[0].url, "./" + promt + ".png")
    return "./" + promt + ".png"

