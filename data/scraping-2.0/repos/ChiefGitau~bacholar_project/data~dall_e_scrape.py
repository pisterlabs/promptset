import shutil
import openai
import requests
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import Counter
import logging

def stop_word_checker(common_word, index = 0):

    for word in common_word:
        if word[0] in stopwords.words('english'):
            return word[0], index
        index += 1


    raise Exception("In the ranking of most common words there are no stopwords that can be eliminated")

def ranking_words(text):
        # split() returns list of all the words in the string
    split_it = text.split()
    # Pass the split_it list to instance of Counter class.
    counter = Counter(split_it)


    return counter.most_common(len(counter))

def wordcount(text, ranks):

    if len(text) <= 1000:
        return text

    most_occur, index = stop_word_checker(ranks)
    text = text.replace(" "+ most_occur+ " ", " ")
    ranks.pop(index)



    return  wordcount(text, ranks)


def char_simplify(text) :
    if len(text) <= 1000:
        return text
    print(text)
    logging.info("---------------")
    logging.info("to many char reducing --> " + str(len(text)))

    text = text.lower().replace(" a ", " ")
    text = re.sub("[,.;\[\]?\"]", "", text)

    text = wordcount(text, ranking_words(text))

    logging.info("new length is -->  " + str(len(text)))
    logging.info("---------------")
    print(text)

    return text


def display(img_url):
    img_url = [image['url'] for image in img_url]
    print(img_url)

def prompt_download(prompts,location_prompt):
   try:
        # print("reading propmts" + prompts)
        prompt_list = []
        for prompt in prompts:
            print("reading prompts at " + str(location_prompt) + str(prompt))
            f = open(location_prompt + prompt, 'r')
            text = f.read().replace('\n', ' ').replace('\r', ' ')
            prompt_list.append(char_simplify(text))


        if  len(prompt_list) > 0:
            return prompt_list
   except:
       print("failed to read prompts")

def generate_image(prompt, number, size):
    try:
        response = openai.Image.create(
            # model = "code-davinci-002",
            prompt=prompt,
            n=number,
            size= size,
            # response_format="url"
        )
        img_url = response['data']
        img_url = [image['url'] for image in img_url]
        return img_url
    except openai.error.OpenAIError as e:
        print(e.http_status)
        print(e.error)

def download(name_format, images,location_down,  image_counter):

    try:
        for i in tqdm (range(len(images)), desc ="downloading to " + location_down + "...."):
            url = images[i]
            image =requests.get(url, stream= True)

            if image.status_code == 200:
                with open(location_down+name_format+str(image_counter)+".png", 'wb') as f:
                    shutil.copyfileobj(image.raw, f)
                image_counter += 1
            else:
                print("image cant be retrieved")
        #
        # for title in name_format:
        #     with open(location+"{}.jpg".format(title), "wb") as f:
        #         f.write(image.content)
    except:
        print("failed to download")
        return name_format

def general_creation(location, size):
    prompt_text = ['prompt_0.txt', 'prompt_1.txt', 'prompt_2.txt', 'prompt_3.txt', 'prompt_4.txt']


    prompts = prompt_download(prompt_text, location)
    prompt_counter =0

    #
    for prompt in prompts:
        image_counter = 0
        num = size
        while num > 0:
            if num > 10:
                images = generate_image(prompt, 10, '1024x1024')
                num -= 10
                download("dalle_" + str(prompt_counter) + "_", images,
                         "data_images/prompt_" + str(prompt_counter) + "/",image_counter)
                image_counter += 10

            else:
                images = generate_image(prompt, num, '1024x1024')
                download("dalle_" + str(prompt_counter) + "_", images,
                         "data_images/prompt_" + str(prompt_counter) + "/",image_counter)
                num = 0
        prompt_counter += 1






if __name__ == "__main__":
    nltk.download('stopwords')
    openai.api_key = "ADD KEY"
    size = 100
    location = 'prompts/'

    general_creation(location, size)



