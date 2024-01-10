from openai import OpenAI
import openai
client = OpenAI()
import os
from googlesearch import search
import requests



def is_valid_link(url):
    """
    Check if a given URL is accessible and valid.

    :param url: URL to be checked.
    :return: True if the URL is valid, False otherwise.
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def get_valid_links(query, num_links=4):
    """
    Performs a Google search and retrieves a specified number of valid links.

    :param query: The search query.
    :param num_links: Number of links to retrieve.
    :return: A list of valid URLs.
    """
    try:
        links = search(query, num_results=num_links)
        valid_links = [link for link in links if is_valid_link(link)]
        return valid_links
    except Exception as e:
        print(f"An error occurred while retrieving links: {e}")
        return []

def generate_blog_post(topic, audience, tone, bLength):
    """
    Generates a blog post using OpenAI's GPT-3.5-Turbo or GPT-4.

    :param topic: The topic of the blog post.
    :param audience: The target audience for the blog post.
    :param tone: The desired tone of the blog post.
    :param bLength: Desired blog length in words.
    :return: A string containing the generated blog post.
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OpenAI API key not found in environment variables.")
        return None

    valid_links = get_valid_links(topic, 4)
    print(valid_links)

    initial_directive = f"""
    Generate a comprehensive and engaging blog post about '{topic}' suitable for {audience}. The post should be at least {bLength} words long, written in a {tone} tone. Ensure the post includes:

    - A compelling introduction that hooks the reader.
    - A series of informative sections with clear subheadings.
    - Practical examples or case studies.
    - Calls to action throughout the post.
    - A conclusion that summarizes the main points and encourages further engagement.
    - Use the following related links as references and cite them appropriately in the post: {', '.join(valid_links)}.
    - be sure to list the links at the end of the post if they do not show up somewhere else: {', '.join(valid_links)}.

    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # or "gpt-4" based on your preference
            messages=[
                {"role": "system", "content": "You are a professional freelance copy writer, with the capabilities of an expert webmaster as well not to mention you hold a doctorate in psychology, mental health well being, physiology and energy work"},
                {"role": "user", "content": initial_directive},
            ])

        if response.choices:
            return response.choices[0].message.content.strip()
            
        else:
            print("No content returned in the response.")
            return None
    except Exception as e:
        print(f"An error occurred while generating the blog post: {e}")
        return None