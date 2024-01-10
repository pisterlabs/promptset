import base64
import random
import os
import requests
from openai import OpenAI
from skpy import Skype

def send_reminder(event, context):
     # retrieve keys
     openai_key = os.environ.get('OPENAI_KEY')

     # Initialize OpenAI API key
     client = OpenAI(api_key = openai_key)

     # Connect to Skype with environment variables
     sk_username = os.environ.get('sk_username')
     sk_password = os.environ.get('sk_password')
     sk = Skype(sk_username, sk_password)

     # functions
     def download_image(url, filename):
          # Send an HTTP GET request to the URL
          response = requests.get(url)
     
          # Check if the request was successful
          if response.status_code == 200:
               # Open a file in binary write mode
               with open(filename, 'wb') as file:
                    # Write the content of the response to the file
                    file.write(response.content)
               print(f"Image downloaded successfully: {filename}")
          else:
               print("Failed to download the image")


     # define some coffee words
     coffee_words = [
     "espresso", "latte", "cappuccino", "arabica", "robusta", 
     "mocha", "americano", "macchiato", "ristretto", "frappuccino",
     "barista", "brew", "caffeine", "caf", "filter",
     "grinder", "roast", "aroma", "crema", "drip",
     "milk", "sugar", "syrup", "whipped cream", "chocolate",
     "caramel", "vanilla", "hazelnut", "cup", "mug",
     "thermos", "beans", "grounds", "shot", "foam",
     "latte art", "kettle", "scale", "pour over", "steamed milk",
     "biscotti", "pastry", "caffeine", "coffeehouse", "flavor"
     ]

     # define some dirty words
     dirty_words = [
     "grimy", "soiled", "unclean", "stained", "muddy",
     "dusty", "filthy", "smudged", "tarnished", "spotted",
     "sullied", "contaminated", "polluted", "greasy", "messy",
     "mucky", "grubby", "dingy", "grungy", "smeared",
     "tainted", "discolored", "cloudy", "murky", "foul",
     "sloppy", "untidy", "cluttered", "littered", "scummy",
     "icky", "nasty", "gross", "unwashed", "scruffy",
     "gunky", "slovenly", "bedraggled", "besmirched", "blotchy",
     "bemired", "crusty", "defiled", "dreggy", "fetid",
     "gloppy", "miasmic", "putrid", "rank", "sludgy"
     ]

     # define words of glory
     glory_fame_words = [
     "accolade", "honor", "fame", "prestige", "reputation",
     "celebrity", "renown", "applause", "tribute", "kudos",
     "laurels", "acclaim", "recognition", "distinction", "eminence",
     "glory", "success", "triumph", "notoriety", "respect",
     "dignity", "admiration", "esteem", "veneration", "praise",
     "commendation", "fame", "stardom", "limelight", "heroism",
     "nobility", "majesty", "grandeur", "magnificence", "splendor",
     "brilliance", "radiance", "notability", "acknowledgment", "celebration",
     "popularity", "fame", "repute", "awards", "merit",
     "prominence", "lionization", "ovation", "exaltation", "panegyric"
     ]

     # define artistic styles
     artistic_styles = [
     "Photorealistic", "Cartoon", "Futuristic",  "Surrealistic",    "Impressionistic",
     "Abstract",  "Pixel Art",  "Watercolor",
     "Oil Painting",  "Charcoal Sketch",  "Minimalist",   "Gothic",
     "Art Deco",  "Cubism",  "Steampunk",  "Anime",
     "Graffiti",   "Pop Art",   "Art Nouveau"
     ]

     # sample from both words and put together in an array
     coffee_word_sample = random.sample(coffee_words, 1)
     dirty_word_sample = random.sample(dirty_words, 1)
     coffee_dirty_words_sample = f'{coffee_word_sample} and {dirty_word_sample}'

     # sample from glory words
     glory_fame_words_sample = random.sample(glory_fame_words, 1)

     # sample from artistic styles
     artistic_styles_sample = random.sample(artistic_styles, 1)

     # define system message
     system_msg = "You are a bot which sends out short, friendly but funny reminders into a group chat at wednedays 4 pm to clean the coffee machine and finds a person to volunteer."

     # define user message
     user_msg = f'Please generate a very short reminder to clean the coffee machine. Please use these two words somewhere in the sentence to refer to a dirty machine: {coffee_dirty_words_sample}. Make the volunteering more attractive by describing it with {glory_fame_words_sample}. Remember: Be short!'

     # Make the API request
     message_response = client.chat.completions.create(
     model="gpt-4-1106-preview",
     messages=[
          {"role": "system", "content": system_msg},  
          {"role": "user", "content": user_msg}
     ]
     )

     # Extract the actual response text
     message_response_text = message_response.choices[0].message.content


     # generate image
     img_response = client.images.generate(
          model="dall-e-3",
          prompt=f'A very {dirty_word_sample} and {random.sample(dirty_words, 1)} looking coffee machine used to make {coffee_word_sample} coffee in a {artistic_styles_sample} style hightlighting how {random.sample(dirty_words, 1)} and dirty looking it is and a happy chicken cleaning it.',
          size="1024x1024",
          quality="standard",
          n=1
     )

     # define url
     image_url = img_response.data[0].url

     # Path where the image will be saved
     file_path = "/tmp/img.png"

     # download image
     download_image(image_url, file_path)

     # Print response
     print(message_response_text)

     # store chats 
     # ch = sk.contacts["live:.cid.6cace4c66e5c4e19"].chat
     ch = sk.chats["19:f001a4cf98cd4e0d877fc54006586918@thread.skype"]

     # send image
     with open("/tmp/img.png", "rb") as f:
          ch.sendFile(f, "img.png", image=True)

     # send message
     ch.sendMsg(message_response_text)


