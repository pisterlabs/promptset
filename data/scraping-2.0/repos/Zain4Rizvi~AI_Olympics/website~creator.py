#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Necessary modules
from io import BytesIO
import requests         # pip install requests
from PIL import Image   # pip install pillow
import openai    # pip install openai
openai.api_key = "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"


# this function returns a dictionary object which contains a song's title, lyrics, and cover image url

def generate_song(genre, topic):
    text_prompt = f"Act as artist: Write lyrics for a song of this genre with 2-4 verses of the {genre} genre with the topics of {topic}"
    chatgpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text_prompt}],
        temperature=0.9,
        max_tokens=2000,
        top_p=0.95)

    response = chatgpt_response['choices'][0]['message']['content'].strip()

    text_prompt = "make a title for these lyrics: " + response

    chatgpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text_prompt}],
        temperature=0.9,
        max_tokens=2000,
        top_p=0.95)

    title = chatgpt_response['choices'][0]['message']['content'].strip()

    return {'title': title, 'lyrics': response}


def generate_album_cover(topics):
    # use the chatgpt_response to get a randomized combimed topic for the album cover based on the all topics
    text_prompt = "Find a 5-word generalized topic that connects all of the following topics:" + topics
    chatgpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text_prompt}],
        temperature=0.9,
        max_tokens=2000,
        top_p=0.95)
    # update topics
    topics = chatgpt_response['choices'][0]['message']['content'].strip()
    print(topics)
    # text_prompt3 = input("Describe your album cover in one word\n")
    pre_img_prompt = """Describe a """ + \
        """wordless album cover with topic: """ + topics + "in 20 words"

    pre_img_responses = openai.Completion.create(
        model="text-davinci-003",
        prompt=pre_img_prompt,
        temperature=0.1,
        max_tokens=300,
        top_p=0.88,
        best_of=1,
        frequency_penalty=0.2,
        presence_penalty=0)

    image_prompt = pre_img_responses['choices'][0]['text'].strip()
    image_object = openai.Image.create(
        prompt=image_prompt,
        n=1,
        size="512x512")

    image_url = image_object['data'][0]['url']

    print(f"url: {image_url}")

    return image_url


# In[2]:


# # Use ChatGPT
# num_songs = int(input("How many songs?"))
# song_lst = []
# topics = ""
# # generate responses for the num_songs of vared inputs
# for i in range(num_songs):
#     print("\nSong " + str(i + 1))
#     text_prompt1 = input("What genre of music for?\n")
#     text_prompt2 = input("What topic?\n")
#     text_prompt = "Act as artist: Write lyrics for a song of this genre with 2-4 verses of the" + \
#         text_prompt1 + "genre with the topics of" + text_prompt2
#     chatgpt_response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": text_prompt}],
#         temperature=0.9,
#         max_tokens=2000,
#         top_p=0.95)

#     # Get the title of thee songs
#     response = chatgpt_response['choices'][0]['message']['content'].strip()
#     song_lst.append(response)

#     text_prompt = "make a title for these lyrics: " + response
#     chatgpt_response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": text_prompt}],
#         temperature=0.9,
#         max_tokens=2000,
#         top_p=0.95)

#     title = chatgpt_response['choices'][0]['message']['content'].strip()

#     # print response for every single output of i
#     print("\nTitle of the song: ")
#     print(title)
#     print(response)
#     # update topics acc by adding new abtained strs
#     topics = topics + ", " + text_prompt2


# # use the chatgpt_response to get a randomized combimed topic for the album cover based on the all topics
# text_prompt = "Find a 5-word generalized topic that connects all of the following topics:" + topics
# chatgpt_response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": text_prompt}],
#     temperature=0.9,
#     max_tokens=2000,
#     top_p=0.95)
# # update topics
# topics = chatgpt_response['choices'][0]['message']['content'].strip()
# print(topics)
# text_prompt3 = input("Describe your album cover in one word\n")


# # In[3]:


# # Create image prompt yourself
# image_prompt_ = """Describe artistic realistic illustration of apple pie"""

# # Let AI create image prompt

# pre_img_prompt = """Describe a """ + text_prompt3 + \
#     """wordless album cover with topic: """ + topics + "in 20 words"

# pre_img_responses = openai.Completion.create(
#     model="text-davinci-003",
#     prompt=pre_img_prompt,
#     temperature=0.1,
#     max_tokens=300,
#     top_p=0.88,
#     best_of=1,
#     frequency_penalty=0.2,
#     presence_penalty=0)

# image_prompt = pre_img_responses['choices'][0]['text'].strip()

# print(image_prompt)


# # In[4]:


# # Use Dall-E-2
# image_object = openai.Image.create(
#     prompt=image_prompt,
#     n=1,
#     size="512x512")

# image_url = image_object['data'][0]['url']


# # In[5]:


# # See Image
# url_response = requests.get(image_url)
# image = Image.open(BytesIO(url_response.content))
# image


# # In[6]:


# # Save image as a jpg file
# name = 'new_image 2'
# image_name = name + '.jpg'

# if url_response.status_code == 200:
#     with open(image_name, "wb") as f:
#         f.write(url_response.content)
#         print("\033[1;36m Image saved successfully")    # Color print code!
# else:
#     print("Failed to download image")


# # In[ ]:
