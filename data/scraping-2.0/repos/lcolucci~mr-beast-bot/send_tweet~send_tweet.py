import playwright
import  time
import json
from bs4 import BeautifulSoup, Tag
from random import shuffle

# import send_tweet.initial_msg as initial_msg
# from langchain.chat_models import ChatOpenAI
# from langchain.schema import SystemMessage, BaseMessage, AIMessage


from playwright.sync_api import Playwright, sync_playwright, expect

INITIAL_MSG = """
@{handle} I am a MR_BEAST_AGENT!!! I was built as part of the @agihouse_org Hackathon on 07/22. 

My goal is to find one MR.BEAST fan and give them $50. 

All you need to do is REPLY to 1) tell me a good deed you did in the last week and 2) your Venmo handle.
"""

class TwitterBot:

    def __init__(self, master_list_path: str): 

        self.master_list = self.load_handles(master_list_path)




    def save_current_state(self, path):

        # Serializing json
        json_struct = json.dumps(self.master_list, indent=4)
        
        # Writing to sample.json
        with open(path, "w") as outfile:
            outfile.write(json_struct)


        

    def get_next_handle(self):

        handle = None
        for i, follower in enumerate(self.master_list):
            if follower['sent_message'] == 'False':
                self.master_list[i]['sent_message'] = 'True'
                self.save_current_state('./send_tweet/temp.json')
                handle = follower['nametag']
                break

        if handle == None:
            print('NO MORE HANDLES!!!')
        return handle
    
    def load_handles(self, path):

        followers = None
        if '.csv' in path:
            with open(path, 'r') as file:
                handles = file.readlines()

            handles = [handle.strip() for handle in handles]
            
            followers = []
            for i, handle in enumerate(handles): 
                if i == 0: 
                    continue
                
                split_line =  handle.split(',')

                follower = {
                    'nametag': split_line[0],
                    'profile_link': split_line[1],
                    'verified': split_line[2],
                    'sent_message': split_line[3]
                }
                followers.append(follower)

        elif '.json' in path:
            with open(path, 'r') as file:
                followers = json.load(file)

        return followers

         
    


def auto_login(playwright: Playwright, auth_json_path=None): 
    '''uses cookies to loigg'''

    url = 'https://twitter.com/home'

    browser = playwright.chromium.launch(headless=False, timeout=60000)
    context = browser.new_context(storage_state=auth_json_path)
    page = context.new_page()
    page.goto(url)

    return browser, page
    

def manual_login(playwright: Playwright, auth_json_path=None): 

    url = 'https://twitter.com/i/flow/login'
    username = 'mr_beast_bot'
    password = 'P6yKTy0Q4ZJ#7Mml'
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(storage_state=auth_json_path)
    page = context.new_page()
    page.goto(url)
    page.get_by_label("Phone, email, or username").click()
    page.get_by_label("Phone, email, or username").fill(username)
    page.get_by_role("button", name="Next").click()
    time.sleep(3)
    page.get_by_label("Password", exact=True).click()
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_test_id("LoginForm_Login_Button").click()
    time.sleep(3)
    page.goto('https://twitter.com/home')
    
    return browser, page


def scrape_webpage(page, taraget_url, file_name='target.html'):

    page.goto(taraget_url)
    time.sleep(3) # just to make sure things load
    html = page.content()
    # save html
    with open(file_name, 'w') as file:
        file.write(html)



def create_and_send_tweet(page, handle: str):

    default_msg = INITIAL_MSG

    customized_msg = default_msg.format(handle=handle)

    page.get_by_test_id("tweetTextarea_0").fill(customized_msg)
    time.sleep(3)
    page.get_by_test_id("tweetButtonInline").click()
    time.sleep(3)
    page.goto('https://twitter.com/mr_beast_bot')
    time.sleep(3)
    
    html = page.content()
    soup = BeautifulSoup(html, 'html.parser')
    tweets_as_bs4 = soup.find_all("article",{"data-testid": "tweet"}) 
    for t in tweets_as_bs4[0]: 
        links = t.find_all("a", {"href": True}) #["href"]
        results = []
        for link in links: 
            if link["href"].startswith("/mr_beast_bot/status"):
                results.append(link["href"]) #finds both tweet and tweets/analytics links
        results = results[0] #get first link only (only the tweet)
    
    link_to_tweet = "https://twitter.com" + results
    return page, link_to_tweet


def send_initial_tweet(handle: str): 

    with sync_playwright() as playwright:
        browser, page = manual_login(playwright, auth_json_path=None)

        page, link_to_tweet = create_and_send_tweet(page, handle)


    return page, link_to_tweet

if __name__ == "__main__":

    master_list_path = './send_tweet/master_json_test.json'
    twitter_bot = TwitterBot(master_list_path=master_list_path)

    handle = twitter_bot.get_next_handle()

    _, link_to_tweet = send_initial_tweet(handle)
    print(link_to_tweet)




