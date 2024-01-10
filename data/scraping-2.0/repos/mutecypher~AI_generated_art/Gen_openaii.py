import openai
import webbrowser
import api
import urllib.request as ulr
import os
import ssl

OPENAI_API_KEY = api.open_key()
openai.api_key = OPENAI_API_KEY
ssl._create_default_https_context = ssl._create_unverified_context


print()
print()
print("Welcome to the OpenAI Image Generator!")


colors = [

    ##, 
    "white",
        "black",
    "grey",
        "brown",
    ##, "spotted"
    ]
animals = [ 
           


            "yorkshire terrier",
            "pomeranian",
            ##"corgie",
            "pooch",            
             "great dane",
            "boxer",
            "poodle",
            "shih tzu",
            "pit bull",
            "maltese",
            "poodle",
            "bulldog",
            "labrador",
            "golden retriever",
            "german shepherd",
            "husky",
            "rottweiler",
            "beagle",
            "dachshund",
            "chihuahua"
            
           ]
poses = [ "facing forward on deck",
          "facing forward on lawn", "facing forward in garage"
         ]
mouths =[
    ##"mouth closed",  
    ##"lizard in mouth",
    "with a rat in mouth",
    ##"mouse in mouth",
    "with bird in mouth",
    
    ##"closed mouth"
    ##,
    "with a stick in mouth"
    ##"panting"
    ]

for color in colors:
    for animal in animals:
        for pose in poses:
            for mouth in mouths:
                n = 10
                prompt = f"{color} {animal}  {pose} {mouth}"
                folder_path = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/dog_open/'
                image_prefix = str(prompt.replace(" ", "_"))
    ##image_count = 0
                lynxie = openai.Image.create(prompt= image_prefix, n=n, size="256x256")
                print("the prompt this time is " , prompt, "    ")
                for i in range(n):
                    image_url = lynxie['data'][i]['url']
                 ##webby = webbrowser.open_new(image_url)
                    ulr.urlretrieve(lynxie['data'][i]['url'], folder_path + image_prefix + str(i) + '.jpg')


print("completed")





print()
print()
print("Thank you for using the OpenAI Image Generator!")
