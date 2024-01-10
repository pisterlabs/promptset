import json
import re
from dotenv import load_dotenv
import openai
import os
import json

load_dotenv()
api_key = os.getenv('API_KEY')
openai.api_key = api_key

def extract_data(profile_data):
    extracted_data = []
    username = profile_data["username"]
    
    latest_igtv_videos = profile_data["latestIgtvVideos"]
    for video in latest_igtv_videos:
        caption = video["caption"]
        likes = video["likesCount"]
        comments = video["commentsCount"]
        extracted_data.append({"username": username, "caption": caption, "likes": likes, "comments": comments})

    latestPosts = profile_data["latestPosts"]
    for post in latestPosts:
        caption = post["caption"]
        likes = post["likesCount"]
        comments = post["commentsCount"]
        extracted_data.append({"username": username, "caption": caption, "likes": likes, "comments": comments})
    
    return extracted_data

def get_top_data(extracted_data, n):
    sorted_data = sorted(extracted_data, key=lambda x: x["likes"], reverse=True)
    top_n_captions = []
    
    for video in sorted_data:
        caption = video["caption"]
        lowercase_caption = caption.lower()
        if "giveaway" not in lowercase_caption and "sale" not in lowercase_caption:
            cleaned_caption = re.sub(r'[^a-zA-Z0-9\s]', '', caption)
            caption_without_newlines = cleaned_caption.replace('\n', ' ')
            top_n_captions.append(caption_without_newlines)
            if len(top_n_captions) == n:
                break  
    return top_n_captions

def main():
    json_file_path = 'backend/data/insta_data.json'

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    top_captions_per_user = {}

    for profile_data in data:
        extracted_data_per_user = extract_data(profile_data)
        username = extracted_data_per_user[0]["username"] 
        top_captions = get_top_data(extracted_data_per_user, 1)
        if top_captions:
            top_captions_per_user[username] = top_captions[0]
    
    top_caption_paragraph = "\n".join(top_captions_per_user.values())
    openai_data = {
        "model": "text-curie-001",
        "prompt": f"List 2-3 keywords that suggest fashion outfit items from this para :\n\n{top_caption_paragraph}",
        "max_tokens": 50        
    }
    response = openai.Completion.create(**openai_data)
    generated_text = response.choices[0].text.strip()
    return generated_text

if __name__ == "__main__":
    top_caption = main()
