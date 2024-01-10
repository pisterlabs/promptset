#Import the required modules
import openai #needed for error handling 
from openai import OpenAI #OpenAI is a class or module from the openai library that allows us to use the openai api
import os #needed for interacting with the operating system level functionality
from flask import Flask, request #needed for creating RESTful APIs, handling https requests and responses in a server. So we import main Flask class and request object
import requests #needed for making https requests to communicate with web servers e.g fetching data from external APIs
import tweepy


from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv()) #read the local .env file

#Authenticate to the OpenAI API
#create an object/instance of the OpenAI class by calling the constructor OpenAI() and assigning it to the client variable
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY") 
    )

#Info needed for the Twitter API authentication
consumer_key=os.getenv("X_API_KEY")
consumer_secret=os.getenv("X_API_SECRET_KEY")
access_token= os.getenv("X_ACCESS_TOKEN")
access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET_KEY")

#Set the Model Names and Context
TEXT_MODEL = "gpt-3.5-turbo"
IMG_MODEL= "dall-e-2"
context = [{'role':'system', 'content': """You are a friendly AI assistant that helps 
    compose professions-sounding tweets for twitter that often go viral based on a website I provide.
    You will provide a summary of the website in 5 words or less."""}]


#Get the summary for the website 
def get_summary(website, temperature=0): #Set website and temperature as parameters
    prompt = "Please summarize this website " + website
    print(prompt)

    """make a request to the openai api by calling the create method on the completions attribute of the chat attribute of the client object.
        response is a json object and we interested in the choices field which is an array.
        Since the number of responses, n = 1, we only have one response and therefore we use index 0 to access this.
        In this object accessed, we have 2 fields/keys which are index and message. We want message.
        Message is an object, and we have two keys. Role and content. We want the content
    """

    try:
        response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=context,
        temperature=temperature
    )
        return response.choices[0].message.content

    #The object openai.APIError has the fields http_status and error that we can access using the dot operator  
    except openai.APIError as e:
        print(e.http_status)
        print(e.error)
        return e.error


#Create an AI Image based on the website summary
def generate_image(summary):
    print(summary)

    """make a request to the openai api by calling the generate method on the images attribute of the  the client object.
        response is a json object and we interested in the data field which is an array.
        Since the number of responses, n = 1, we only have one response and therefore we use index 0 to access this.
        In this object accessed, we have many fields/keys and but we want url.
    """

    try:
        response = client.images.generate(
            prompt=summary,
            model=IMG_MODEL,
            size="1024x1024",
            n=1,
            quality="standard"
        )
        image_url = response.data[0].url
        return image_url
    
    except openai.APIError as e:
        print(e.http_status)
        print(e.error)
        return e.error


#Download the image in binary form and store it in a file in the directory
def download_image(imageURL):
    print("downloading - ", imageURL)
    img_data = requests.get(imageURL).content #sends a get request to the specified url using the requests module, and .content retrieves the content of the response which is the binary data of the image
    with open("dalle_image.jpg", "wb") as handler:
        handler.write(img_data)

    return "dalle_image.jpg"


#Upload the image media using Version 1 of Twitter Api

def upload_image(image):
    """"tweepy.OAuth1UserHandler is a class that takes in the api and consumer keys/secret keys for authentication.
    Therefore auth is a specific instance of the OAuth1UserHandler class and it is specific because it has specific credentials used to initialize it.
"""
    auth  = tweepy.OAuth1UserHandler(
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret
    )
    
    #Create an instance of the API class from Tweepy. The instance api is configured or initialized with the authentication information provided by the auth allowing interaction the Twitter's APi on behalf of the authenticated user
    api = tweepy.API(auth)

    #call the media_upload method on the api object. This method is for uploading media files and takes the filename of the media file as a parameter
    media = api.media_upload(filename=image)

    #returns information about the uploaded media such as its ID, URL and other details
    return media


#Send Tweet using Version 2 of Twitter API
def send_tweet(summary, image):
    #create an instance client of the tweepy.Client class and initialize/configure it with the passed parameters which are twitter api credentials. 
    #the object created interacts with the twitter api
    client = tweepy.Client(
        consumer_key=consumer_key, consumer_secret=consumer_secret,
        access_token=access_token, access_token_secret=access_token_secret
    )

     #upload image to Twitter servers and get the media metadata
    media = upload_image(image)
    media_ids = [media.media_id]

    #send the tweet
    response = client.create_tweet(
        text=summary,
        media_ids=media_ids
        )
    
    #prints the URL of the created tweet to the console. The tweet ID is obtained from the response.
    print("https://twitter.com/user/status/{}".format(response.data['id']))

#Create an instance app of the Flask class and pass in the __name__ parameter
app = Flask(__name__)   

#create a decorater for defining a route for handling HTTP GET requests at the specified URL. The function within this decorater is the one called when the url is called
@app.route('/tweets', methods=['GET'])

def index():

    #retrieve arguments: request is an object in Flask that represents the incoming HTTP request
    #request.args is a dictionary-like object containing the parsed query parameters from the URL e.g service, id, index
    args = request.args 
    print(args) #debugging
    service = args["service"]

    #get summary of website url
    summary = get_summary("http://www.amazon.com/" + service) 
    print(summary) 

    #generate an image using the summary
    image_name = download_image(generate_image(summary))
    print(image_name) #debugging
    
    #tweet the image
    send_tweet(summary, image_name)

    return 'Tweet sent!'

app.run(port=5000)  # run app in debug mode on port 5000
