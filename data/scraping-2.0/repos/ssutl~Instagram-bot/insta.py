from instagrapi import Client
import requests
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from typing import Dict
import textwrap
from dotenv import load_dotenv
import os
import openai
import random
import schedule
import time
import json
load_dotenv()

# Get the environment variables
insta_username = os.getenv('insta_username')
insta_password = os.getenv('insta_password')
kton_username = os.getenv('kton_username')
kton_password = os.getenv('kton_password')
openai.api_key = os.getenv('openAI_key')



def getQuote(index):
    with open("quotesList.json", "r") as jsonFile:
        json_data = json.load(jsonFile)
        if index >= len(json_data):
            return None
        return json_data[index]
        
def getImageD():
    ##Using Dall-e
    
    # Generate an ultra-realistic anime cityscape that immerses the viewer in a bright and futuristic metropolis. The attention to detail is paramount – from the intricately designed skyscrapers with realistic glass reflections to the individual leaves swaying on the holographic trees. Every aspect of the scene should evoke a sense of realism and wonder.
    
    #Action photography of a parkour athlete jumping between urban structures, using a fast shutter speed.
    
    # Lifestyle photography of someone listening to vinyl records, using warm tones to evoke nostalgia.
    
    # Lifestyle photography of a black 80s DJs playing music and mixing vinyls with his crew, using warm tones to evoke nostalgia.
    
    #Lifestyle photography of the 80s streets with black people. DJs playing music and mixing vinyls. Kids running. Palm trees. using warm tones to evoke nostalgia.
    
    #Lifestyle photography of the 80s streets with black people. DJs passionately mixing vinyl records on turntables, where the vinyl decks themselves are miniature cityscapes, complete with intricate details. Kids running. Palm trees. using warm tones to evoke nostalgia.
    
    
    
    try:
        response = openai.Image.create(
        prompt="Lifestyle photography of the 80s streets with black people.Vibrant. DJs passionately mixing vinyl records on turntables, where the vinyl decks themselves are miniature cityscapes, complete with intricate details. Using warm tones to evoke nostalgia.",
        n=1,
        size="1024x1024"
        )
        imageUrl = response['data'][0]['url']
        ##Saving the file
        response = requests.get(imageUrl)
        with open('image.jpg', 'wb') as f:
            # Write the contents of the response to the file
            f.write(response.content)
            
    except openai.error.OpenAIError as e:
        print(f'Request failed: {e}')

def getImageU():
    ##Requests Unsplash
    random_url="https://api.unsplash.com/photos/random"
    access_key = "QyIVMq6A6fL2y7WlNE9XsU2X7F40JUSTj-nsCaX_MYI"
    headers = {"Authorization": f"Client-ID {access_key}"}
    params = {'query': 'modern building black', 'orientation': 'squarish'}
    
    try:
        unsplash_response = requests.get(random_url,headers=headers,params=params)
        unsplash_response.raise_for_status() #Anything thats not 200
        random_image = unsplash_response.json()["urls"]["raw"]
        
        ##Saving the file
        response = requests.get(random_image)
        with open('image.jpg', 'wb') as f:
            # Write the contents of the response to the file
            f.write(response.content)
            
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')

def createPost(index):
        # Open image
    img = Image.open("image.jpg")
    draw = ImageDraw.Draw(img, 'RGBA')
    font_size = 1
    font = ImageFont.truetype("font.ttf", font_size)
    
    # Get quote information
    quote = getQuote(index)
    global title, author
    title, author = quote['Title'], quote['Author']
    text = quote['Text']
    global caption
    caption=f'Quote extracted from {author.replace(";"," & ")}\'s "{title}" {randomEmoji()} \n #Quotes #Books #HumblySubtly #MentalMobility'
    
    
    # Set background color
    bg_color = (0, 0, 0, 200)  # Black color with 70% opacity
    

    # Wrap text and calculate total height
    wrapped_text = textwrap.wrap(text, width=40) #Maximum 20 characters per line, splits into array of strings
    line_height = font.getsize('hg')[1] #random string to get rough height of a single line (returns a tuple of (height,width))
    total_height = len(wrapped_text) * line_height #jsut multiply each line by their heights
    
    #Find the longest string in wrapped text and continually increase font until it reaches max
    longest_string = max(wrapped_text, key=len)
    
    while font.getsize(longest_string)[0] < 0.8*img.size[0]:
        font_size+=1
        font = ImageFont.truetype("font.ttf", font_size)
        line_height = font.getsize('hg')[1] * 2
        total_height = len(wrapped_text) * line_height
    
        
    # the y-coordinate of the starting point of the text, 
    # which is the point where the text will be drawn on the image.
    y = (img.height - total_height) / 2
    
    
    
    # Draw each line of wrapped text on the image
    #In computing vertical axis goes from zero at top to image height at bottom !
    for line in wrapped_text:
        # Center horizontally
        line_width = font.getsize(line)[0]
        
        #the horizontal position of the starting point of the text, 
        # if the text is horizontally centered within the image.
        line_x = (img.width - line_width) / 2
        
        # Draw background rectangle (defining top left and bottom right point) first line we add padding of 10
        bg_x1, bg_y1 = line_x - 20, y - 10
        bg_x2, bg_y2 = line_x + line_width + 20, y + line_height + 10 #bottom right
        

        # Draw background rectangle and text
        draw.rectangle((bg_x1, bg_y1, bg_x2, bg_y2), fill=bg_color)
        
        # Calculate vertical position for text (to center it within the rectangle)
        bg_center_y = (bg_y1 + bg_y2) / 2
        text_y = bg_center_y - (font.getsize(line)[1] / 2)
    
        draw.text((line_x, text_y), line, font=font, fill=(255, 255, 255))
        
        # To move the y coordinate to the vertical position below previous line
        y += line_height + 20
    
    #Draw rectangle bottom right
    
    
    # Save modified image
    img.save("overlay.jpg")
    
def randomEmoji():    
    EmojiArray = ["📚","🧠","🥭","⌛","♾️","📜","🎯"]
    randomEmojis = random.sample(EmojiArray,2)
    return " ".join(randomEmojis)

def postFunction():
     global current_index
     print("Uploading Post")
     quote = getQuote(current_index)
     if quote is not None:
        # Extract quote data
        quote_text = quote["Text"]
        quote_author = quote["Author"]
        quote_title = quote["Title"]
        
        # Create post
        getImageD()
        createPost(current_index)
        cl.photo_upload('overlay.jpg', caption, extra_data={
            "like_and_view_counts_disabled": True,
            "disable_comments": True
        })
        print(f"Posted: {quote_text} - {quote_author} ({quote_title})")
        current_index += 1  # Increment index for next post
     else:
        print("No more quotes to post")
    
    

testing = input("Are you testing the software?")

if testing == "yes" or testing == "YES" or testing == "Y" or testing == "y":
    imageGeneration = input("Do you want to use DALLE (D) or no (any key)?")
    
    if imageGeneration == "D" or imageGeneration == "d":
        getImageD()
    else:
        getImageU()   
    createPost(28)
else:
    cl = Client()
    cl.login(username=insta_username, password=insta_password)
    #When code starts start from this index
    current_index = 0
    schedule.every().day.at("04:00").do(postFunction)

    
    while True:
        schedule.run_pending()
        time.sleep(1)



    