import json
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from elevenlabs import generate, save
import openai
from nanoid import generate as generate_id
openai.api_key = "add your key"
client = MongoClient("mongodb://localhost:27017/")
db = client["aisplain"]
from bson.objectid import ObjectId
option = webdriver.FirefoxOptions()
option.headless = True
option.add_argument("--headless=new")
driver = webdriver.Firefox(options=option)
from elevenlabs import set_api_key

def getText(url):
    driver.get(url)
    time.sleep(2)
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    body = driver.find_element(By.TAG_NAME, "body").text
    return body


def getSummary(url):
    prompt = '''
    You are a summariser.
    You are given a text from a HTML website and you have to summarise it.
    The text is: "''' + getText(url) + '''"

    Your summary must contain what exactly the website is about, how it works, and anything else you think is important.
    Ignore all call to actions. Just focus on what the website is about.
    '''
    # create a chat completion
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]).choices[0].message.content

    return summary


async def invoke_landingpage_agent(project_id: str, url: str, tone: str):
    print("Landing Page Agent invoked")
    htmlText = getText(url)
    sectionPrompt = '''
    You are representative of a company. 
    Company Summary: ''' + getSummary(url) + ''' 
    You are given all the text from the HTML website of the company.
    The text is: "''' + htmlText + '''"
    Generate:
    A sales pitch for a customer who is on the landing page of the company website.

   
    The tone of the sales pitch:
    '''+tone+'''
    Rules:
    1. You must explain and pitch every section of the landing page that is important to the customer. Don't repeat the same thing.
    2. You must respond with the following JSON object format.
    JSON object format -
    [{section_pitch: "should contain the pitch", target_section_text: "should contain the line of the target text of the section, just one line "}]
    3. Do not respond with anything but the JSON
    4. Do not mention the titles of the sections in the pitch

    '''
    obj = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": sectionPrompt}]).choices[0].message.content
    jsonobj = json.loads(obj)
    for i in jsonobj:
        set_api_key("add your key")
        audio = generate(
            text=i["section_pitch"],
            voice="Valley",
            model="eleven_monolingual_v1"
        )
        fname = generate_id()
        save(audio, "./quickaudio/"+fname+".mp3")
        i["audio"] = fname+".mp3"
    query = {"_id": ObjectId(project_id)}
    newvalues = {"$set": {"landingPageContent": jsonobj}}
    temp = db.projects.update_one(query, newvalues)
    print("Landing Page Agent : Task Complete")

