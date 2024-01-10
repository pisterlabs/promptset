from openai import OpenAI # pip install openai
import requests, json # pip install requests
import jwt	# pip install pyjwt
from datetime import datetime as dt
from io import BytesIO  
from unsplash.api import Api # pip install python-unsplash
from unsplash.auth import Auth
import random
import tempfile
import os

# OPENAI keys
client = OpenAI() # openai client instance
client.api_key = "YOUR CHATGPT KEY"  # YOUR CHATGPT API KEY 

# Unsplash API keys
client_id = "YOUR CLIENT ID" # this is Access Key from your app's settings in your Unsplash profile (I also wrote this in README)
client_secret = "YOUR CLIENT SECRET" # this is Secret key from your app's settings in your Unsplash profile
redirect_uri = "YOUR REDIRECT URI" # this is Redirect URI from Redirect URI & Permissions section


# Ghost API keys
ghost_instance_name = 'play_ghost' # you can name it how do you like
ghost_url = 'YOUR GHOST INSTANCE' # this is URL of your Ghost website
AdminAPIKey = 'YOUR AdminAPIKey' # keys from 'Integrations' tab in settings of your Ghost admin
ContentAPIKey = 'YOUR ContentAPIKey'

# prompts for LLM
prompt_text = "generate one news article in markdown format without images, date or author. it is sufficient to write no more than four paragraphs" # prompt for generating news article in markdown 
prompt_unsplash = 'reduce this sentence to four words' # propmpt for generating query for image search using name of the article
prompt_tag = 'generate one news category for this news article in one word' # prompt for generating news category
words_check = ['your', '[', 'lorem']   # forbidden words -> this helps to find articles that don't contain stuff like [author's name], 'lorem ipsum...' etc.
min_tokens = 200 # minimum number of tokens that answer from AI should contain

class GhostAdmin():
    def __init__(self, siteName):
        self.siteName = siteName
        self.site = None
        self.setSiteData()
        self.token = None
        self.headers = None
        self.createHeaders()

    def setSiteData(self):
        sites = [{'name': ghost_instance_name, 'url': ghost_url, 'AdminAPIKey': AdminAPIKey, 'ContentAPIKey': ContentAPIKey}]
        self.site = next((site for site in sites if site['name'] == self.siteName), None)
        
        return None

    def createToken(self): # creating short-lived single-use JSON Web Tokens. The lifespan of a token has a maximum of 5 minute. You can read more about it: https://ghost.org/docs/admin-api/#token-authentication
        key = self.site['AdminAPIKey']
        id, secret = key.split(':')
        iat = int(dt.now().timestamp())
        header = {'alg': 'HS256', 'typ': 'JWT', 'kid': id}
        payload = {'iat': iat, 'exp': iat + (5 * 60), 'aud': '/v3/admin/'}
        self.token = jwt.encode(payload, bytes.fromhex(secret), algorithm='HS256', headers=header)

        return self.token

    def createHeaders(self):
        if self.site != None:
            self.createToken()
            self.headers = {'Authorization': 'Ghost {}'.format(self.token)}

        return self.headers
    
    def createPost(self, title, body, bodyFormat='html', excerpt = None, tags=None, authors=None, status='draft', featured=False, featureImage=None, slug=None):
        content = {'title': title}
        if bodyFormat == 'markdown': content['mobiledoc'] = json.dumps({ 'version': '0.3.1', 'markups': [], 'atoms': [], 'cards': [['markdown', {'cardName': 'markdown', 'markdown': body}]], 'sections': [[10, 0]]});
        else: content['html'] = body
        if excerpt != None: content['custom_excerpt'] = excerpt
        if tags != None: content['tags'] = tags
        if authors != None: content['authors'] = authors
        content['status'] = status
        content['featured'] = featured
        if featureImage != None: content['feature_image'] = self.site['url']+featureImage
        if slug != None: content['slug'] = slug

        url = self.site['url']+'ghost/api/v3/admin/posts/'
        params = {'source': 'html'}
        result = requests.post(url, params=params, json={"posts": [content]}, headers=self.headers)
        if result.ok: result = 'success: post created (status_code:'+str(result.status_code)+')'
        else: result = 'error: post not created (status_code:' + str(result.status_code) + ')' + str(result.reason)

        return result
    
    def loadImage(self, imagePathAndName):  # creating imageObject for imageUpload function
        try:
            image = open(imagePathAndName, 'rb')
            imageObject = image.read()
            image.close()
            image = BytesIO(imageObject)
            return image
        except Exception as e:
            print(f"An error while loadImage function occurred: {e}")
            return None

    def imageUpload(self, imageName, imageObject):  # uploading image to ghost
        url = self.site['url'] + 'ghost/api/v3/admin/images/upload/'
        files = {"file": (imageName, imageObject, 'image/jpeg')}
        params = {'purpose': 'image', 'ref': imageName}   # 'image', 'profile_image', 'icon'
        result = requests.post(url, files=files, params=params, headers=self.headers)
        return result
    




def download_image_by_id(photo_id, filename): # downloading concrete image from Unsplash site using image ID
    # API endpoint for a specific photo based on ID
    url = f"https://api.unsplash.com/photos/{photo_id}?client_id={client_id}"
    
    # Make a GET request to the Unsplash API for the specific photo
    response = requests.get(url)
    
    if response.status_code == 200:
        # Extract image URL from the response
        image_url = response.json()['urls']['full']
        # print(response.json()['urls'])
        
        # Make a GET request to download the image
        image_response = requests.get(image_url, stream=True)
        # Save the image to a file
        if image_response.status_code == 200:
            with open(filename, 'wb') as file:
                image_response.raw.decode_content = True
                file.write(image_response.content)
            print(f"Image downloaded and saved as {filename}")
        else:
            print("Failed to download image")
    else:
        print("Failed to fetch photo details from Unsplash")

def answer_from_AI(query: str):     # sending question to CHATGPT API
    response = client.chat.completions.create(          
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": query}]
  )
    return response.choices[0].message.content

def unsplash_image_search(query: str):  # sending request for searching picture to Unsplash site
    auth = Auth(client_id, client_secret, redirect_uri) 
    api = Api(auth)
    res = api.photo.random(query=query, orientation='landscape')
    photo_id = res[0].id
    print(res)

    return photo_id
    
def create_temp_directory(): # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    return temp_dir

def delete_all_files_from_temp(temp_dir):
    try:
        # Delete all files in the temporary directory
        for file_name in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file_name)
            os.remove(file_path)
    except Exception as e:
        print(f"An exception occurred: {e}")



if __name__ == '__main__':
   # GENERATING THE NEWS ARTICLE
  generated_text = answer_from_AI(prompt_text)
#   print(f'text: {generated_text[:200]}')

  bad_response = False  

  if len(generated_text.split()) < min_tokens:             # CHECKING FOR VALID AI RESPONSE
    print(f'sorry, less than {min_tokens} tokens')
  else:
      ignore_case_text = generated_text.lower()
      for word in words_check:
          if word in ignore_case_text:
              print('sorry, bad AI response, try again')
              bad_response = True
              break
      if not bad_response:
        title_idx = 0
        found_article = False
        lines = generated_text.split('\n')
        for line in lines:
            title_idx += 1
            if line.startswith('#'):        # finding the line with title
                    article_name = line.replace('#', '').replace('*','').replace('Title', '').replace('title', '').strip() # sometimes there are symbols like '#', '*' and 'title' word in article name
                    found_article = True
                    break
        if found_article == False:
            print('sorry, your LLM generated markdown text without title')
        else:
            generated_body = '\n'.join(lines[title_idx:]) # body of the article shouln't contain its title
            excerpt = str(lines[title_idx + 1].split('.')[0])

            print(f"title: {article_name}")

            # GENERATING QUERY FOR UNSPLASH API 
            query = answer_from_AI(prompt_unsplash + article_name)
            print(f'query for unsplash search: {query}')

            photo_id = unsplash_image_search(query)

            image_name = f'{random.randint(0,999)}.jpg'  # File name to save the downloaded photo
            temp_dir = create_temp_directory()
            output_filename = temp_dir + image_name
            download_image_by_id(photo_id, output_filename)

            # GENERATING TAGS 
            tag = answer_from_AI(prompt_tag + generated_body)
            print(f'tag for article: {tag}')
            post_tags = [{'name': tag }]

            ga = GhostAdmin(ghost_instance_name)	# your Ghost instance
            image_object = ga.loadImage(output_filename) # loading image to Ghost
            result = ga.imageUpload(imageName=image_name, imageObject=image_object)

            num_attempts = 0 # checking for result's validity
            success = False
            while num_attempts <= 3 and (result.status_code >= 500 or result.ok): # if it is server error, try again 3 times (if response is not error, continue)
                    num_attempts += 1
                    if result.ok: 
                        image_url = json.loads(result.text)['images'][0]['url']  # image_url on Ghost server
                        result = 'success: ' + image_url
                        print(result)
                        img_idx = image_url.find('content')
                        image_slug = image_url[img_idx:]     # for POST request that creates a post, image URL should be like: content/images/year/month/image.jpg, so we should get rid of other words in URL
                        create_post = ga.createPost(title=article_name, bodyFormat='markdown', excerpt=excerpt, tags=post_tags,body=generated_body, status='published', featureImage=image_slug)
                        print(f'create_post result: {create_post}')
                        success = True
                        break
                    else: 
                        error_str = 'error: image upload failed (' + str(result.status_code) + ')' + str(result.reason) 
                        print(error_str + '...trying again..')
            if not success and result.status_code >= 500:
                print('I have tryed 3 times, still is not working')
            if not success and result.status_code < 500:
                error_str = 'error: image upload failed (' + str(result.status_code) + ')' + str(result.reason) 
                print(error_str)
            delete_all_files_from_temp(temp_dir)
               

