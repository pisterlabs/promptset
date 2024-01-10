#
# BrightonSEO April 2023, Wordpress GPT with DALL-E2 example.
#

import requests
import json
import base64
import sys
import os
from base64 import b64decode
from pathlib import Path
import openai
from bs4 import BeautifulSoup 
import warnings
warnings.filterwarnings("ignore")

# Configurable options
openai.api_key = "***key here***"
wpUser = "***user name here***"
wpPaassword = "**passwordhere***"
wpURL = 'https://**url here***/wp-json/wp/v2/posts/'
wpImageURL = 'https://**url here***/wp-json/wp/v2/media/'
	
#newsURL = "https://feeds.bbci.co.uk/sport/football/rss.xml"

#newsURL = "http://feeds.bbci.co.uk/news/video_and_audio/science_and_environment/rss.xml"
	
#newsURL = "http://feeds.bbci.co.uk/news/business/rss.xml"
	
#newsURL = "http://feeds.bbci.co.uk/news/health/rss.xml"
	
#newsURL = "http://feeds.bbci.co.uk/news/politics/rss.xml"	
	
#newsURL = "http://feeds.bbci.co.uk/news/technology/rss.xml"
	
newsURL = "http://feeds.bbci.co.uk/news/rss.xml"

DATA_DIR = Path.cwd() / "Dall-E"
DATA_DIR.mkdir(exist_ok=True)

# Create the authentication header for Wordpress
wpCredentials = wpUser + ":" + wpPaassword
wpToken = base64.b64encode(wpCredentials.encode())
wpHeader = {'Authorization': 'Basic ' + wpToken.decode('utf-8'), 'Content-Type': 'application/json'}

print("Fetching BBC XML feed...")

# Fetch the 20 latest news article
xmlResponse = requests.get(newsURL)
xmlSoup = BeautifulSoup(xmlResponse.text, "html.parser")	
newsArticles = xmlSoup.find_all('item')
current_item=0
for item in range(20):
	articleURL = newsArticles[current_item].guid.text
	articleHeadLine = newsArticles[current_item].title.text
	current_item+=1

print("BBC XML feed fetched...")

#print(newsArticles[0].title.text)
#print(newsArticles[0].guid.text)

# Get article array size
articleArraySize = len (newsArticles)

for num in range(1, articleArraySize):
	# Get the first news article in the list, we're not worrying about the others in this example
	articleResponse = requests.get(newsArticles[num].guid.text)
	articleSoup = BeautifulSoup(articleResponse.text, "html.parser")
	articleScrapedContent = articleSoup.find_all('p')
	articleText = ""
	for p_tag in range(len(articleScrapedContent)):
			if p_tag < 0:
				pass
			else:
				articleText = articleText + articleScrapedContent[p_tag].text

	#print(articleText)

	try:
		# Build the completion based on the extracted content for the content
		contentCompletion = openai.ChatCompletion.create(
		  model = 'gpt-3.5-turbo',
		  messages = [
		     {"role": "user", "content": "Rewrite the following text to be more concise: " + articleText}
		  ],
		  temperature = 0
		)

		# Place our new content into the variable wpPostContent
		wpPostContent = contentCompletion['choices'][0]['message']['content']

		# Build the completion based on the extracted content for the content
		titleCompletion = openai.ChatCompletion.create(
		  model = 'gpt-3.5-turbo',
		  messages = [
		     {"role": "user", "content": "Rewrite the following text to be more concise: " + newsArticles[num].title.text}
		  ],
		  temperature = 0
		)

		# Place our new slug into the variable wpSlug, transforming it to be URL safe
		wpSlug = titleCompletion['choices'][0]['message']['content']
		wpSlug = wpSlug.lower()
		wpSlug = wpSlug.replace(" ", "-")

		#print(titleCompletion['choices'][0]['message']['content'])
		#print(wpPostContent)
		#print(wpSlug)

		print ("Generating DALL-E2 image...")
		
		# Create DALL-E2 prompt.
		PROMPT = "Create an image for " + titleCompletion['choices'][0]['message']['content']
		response = openai.Image.create(
		    prompt=PROMPT,
		    n=1,
		    size="1024x1024",
		    response_format="b64_json",
		)

		file_name = DATA_DIR / f"{PROMPT[:5]}-{response['created']}.json"

	  # Save raw JSON response to temporary folder.
		with open(file_name, mode="w", encoding="utf-8") as file:
		    json.dump(response, file)

		# Open JSON response from temporary folder.
		with open(file_name, mode="r", encoding="utf-8") as file:
		    response = json.load(file)

		# Decode and save as PNG file.
		for index, image_dict in enumerate(response["data"]):
		    image_data = b64decode(image_dict["b64_json"])
		    image_file = DATA_DIR / f"{file_name.stem}-{index}.png"
		    with open(image_file, mode="wb") as png:
		        png.write(image_data)
	  
		print("Sending image to Wordpress...")
	  
	  # Send new image to WordPress.      
		data = open(image_file, 'rb').read()
		fileName = os.path.basename(image_file)   
		res = requests.post(url=wpImageURL,
			data=data,
			headers={ 'Content-Type': 'image/jpg','Content-Disposition' : 'attachment; filename=%s'% fileName},
			auth=(wpUser, wpPaassword))
		newDict=res.json()
		newID= newDict.get('id')
		link = newDict.get('guid').get("rendered")
		print (newID, link)
	        
		# Send the new post to Wordpress in a draft state.
		wpData = {
		'title' : titleCompletion['choices'][0]['message']['content'],
		'status': 'publish',
		'slug' : wpSlug,
		'categories': [2],
		'featured_media': newID,
		'content': wpPostContent
		}

		wpResponse = requests.post(wpURL,headers=wpHeader, json=wpData)
		print("Sending post to Wordpress...")
		print(wpResponse)
		
	except Exception as e:
		print("Unable to add post... " + str(e))
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		pass

