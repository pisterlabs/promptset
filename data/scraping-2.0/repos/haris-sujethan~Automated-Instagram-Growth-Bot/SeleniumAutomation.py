from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support import ui
import random
import maskpass

#from secret import UserUsername
#from secret import UserPassword
# The below code gets the username and password manually
# If you would like to import your credentials you can remove the code which prompts you for your username and password


# Since OPENAI no longer offers free credits for their api, the comments are randomly selected from an array of pre written comments.
# If you do have the OPENAI api, implement the following code:

#import openai
#openai.api_key = APIKEY

# output = openai.ChatCompletion.create(
# model="gpt-3.5-turbo",
# messages=[{"role": "user", "content":
# "Generate a generic nice comment to post on instagram, less than 10 words"}]
# )

# Print out the whole output dictionary
# print(output)

# Get the output text only
# print(output['choices'][0]['message']['content'])

# Then replace SendComment.send_keys(random.choice(ListofComments)) with SendComment.send_keys(output['choices'][0]['message']['content'])

instagramURL = 'https://www.instagram.com'
instagramUrlTag = 'https://www.instagram.com/explore/tags/'


# Establishes drivers and opens Instagram
def ChromeDriver():
    try:
        options = webdriver.ChromeOptions()
        options.add_experimental_option('excludeSwitches',
                                        ['enable-logging'])
        global driver
        driver = webdriver.Chrome(options=options,
                                  executable_path=r"C:\chromedriverTesting\chromedriver.exe"
                                  )
        driver.get(instagramURL)
    except Exception as e:
        # If an Exception occurs, prints the error message and quits the driver
        print('An error occured, may be caused by filename, or file location.', e)
        quit()


ChromeDriver()


# Asks the user for their credentials and how many photos they want to interact with

def UserSpecificInformation():
    print('Hello and welcome to the Instagram bot! You should see a Chrome window containing Instagram, please enter your credentials below, and the bot will run.')
    print('\n')
    UserUsername = input('Please Enter Your Instagram Username: ')
    print('\n')
    UserPassword = \
        maskpass.askpass(
            prompt='Please Enter Your Instagram Password: ', mask='*')
    print('\n')
    global HashTag
    HashTag = input('What HashTag Would you Like to Search: #')
    print('\n')
    global Likes
    Likes = int(input('How Many Photos do you Want to interact with? '))
    Username(UserUsername)
    Password(UserPassword)

# Enters the username on Instagram


def Username(UserUsername):
    username = WebDriverWait(driver,
                             20).until(EC.visibility_of_element_located((By.XPATH,
                                                                         "//input[@name='username']")))
    username.send_keys(UserUsername)


# Enters the password on Instagram
def Password(UserPassword):
    password = WebDriverWait(driver,
                             20).until(EC.visibility_of_element_located((By.XPATH,
                                                                         "//input[@name='password']")))
    password.send_keys(UserPassword)


# Presses the login button
def Login():
    login = WebDriverWait(driver,
                          20).until(EC.element_to_be_clickable((By.CSS_SELECTOR,
                                    'button[type="submit"]')))
    login.click()


UserSpecificInformation()

try:
    Login()
except Exception as e:
    print("Login error, please check username and password again")
    print('\n')
    print(e)
time.sleep(3)
print("Login successful")
print('\n')
time.sleep(5)
driver.get(instagramURL)
time.sleep(5)


# Clicks Not now on the instagram pop ups

def IgnorePopUp():
    NotNow = WebDriverWait(driver,
                           20).until(EC.element_to_be_clickable((By.XPATH,
                                                                 "//button[contains(text(), 'Not Now')]")))
    NotNow.click()


IgnorePopUp()
time.sleep(5)
driver.get(instagramUrlTag + HashTag)
# If an error occurs when instgram redirects to the hashtag page, remove driver.get(instagramUrlTag + HashTag) and use:
# def Search(HashTag):
# search = WebDriverWait(driver,
# 20).until(EC.element_to_be_clickable((By.XPATH,
# '//input[@placeholder="Search"]')))
# search.send_keys('#' + HashTag + ' ')
# time.sleep(5)
# search.send_keys(Keys.ENTER)
# time.sleep(5)
# search.send_keys(Keys.ENTER)
# time.sleep(5)
time.sleep(5)

# Counts the amount of photos interacted with
Counter = 0

# Poteinal comments (randomly selected for every post)
ListofComments = ['Great job! Follow me, and I\'ll keep you as a permanent follower',
                  'Well done! Follow me, and I\'ll keep you as a permanent follower',
                  'Impressive! Follow me, and I\'ll keep you as a permanent follower',
                  'Fantastic! Follow me, and I\'ll keep you as a permanent follower',
                  'Bravo! Follow me, and I\'ll keep you as a permanent follower',
                  'Excellent work! Follow me, and I\'ll keep you as a permanent follower',
                  'Terrific! Follow me, and I\'ll keep you as a permanent follower',
                  'Awesome! Follow me, and I\'ll keep you as a permanent follower',
                  'Wonderful! Follow me, and I\'ll keep you as a permanent follower',
                  'Superb! Follow me, and I\'ll keep you as a permanent follower',
                  'Kudos! Follow me, and I\'ll keep you as a permanent follower',
                  'Outstanding! Follow me, and I\'ll keep you as a permanent follower',
                  'You nailed it! Follow me, and I\'ll keep you as a permanent follower',
                  'Keep it up! Follow me, and I\'ll keep you as a permanent follower',
                  'Incredible! Follow me, and I\'ll keep you as a permanent follower',
                  'Remarkable! Follow me, and I\'ll keep you as a permanent follower',
                  'Nice work! Follow me, and I\'ll keep you as a permanent follower',
                  'Thumbs up! Follow me, and I\'ll keep you as a permanent follower',
                  'A+ Follow me, and I\'ll keep you as a permanent follower']


def LikeAndCommentCycle(Likes):
    global Counter
    while Likes > Counter:
        try:
            MainBot()
        except Exception as e:
            print(
                'An error occured, may be caused by no other posts existing in the hashtag')
            print('\n')

            driver.find_element(By.XPATH,
                                "//button[@class='wpO6b  ']//*[@aria-label='Next']"
                                ).click()
            pass


def NextPhoto():
    Next = driver.find_element(By.XPATH,
                               """/html/body/div[2]/div/div/div[3]/div/div/div[1]/div/div[3]/div/div/div/div/div[1]/div/div/div[2]/button""")
    Next.click()


def ImageLike():
    ImageLike = driver.find_element(By.XPATH,
                                    """/html/body/div[2]/div/div/div[3]/div/div/div[1]/div/div[3]/div/div/div/div/div[2]/div/article/div/div[2]/div/div/div[2]/section[1]/span[1]/button""")
    ImageLike.click()


def Follow():
    Follow = driver.find_element(By.XPATH,
                                 """/html/body/div[2]/div/div/div[3]/div/div/div[1]/div/div[3]/div/div/div/div/div[2]/div/article/div/div[2]/div/div/div[1]/div/header/div[2]/div[1]/div[2]/button""")
    Follow.click()


def Comment():
    Comment = driver.find_element(By.XPATH,
                                  """/html/body/div[2]/div/div/div[3]/div/div/div[1]/div/div[3]/div/div/div/div/div[2]/div/article/div/div[2]/div/div/div[2]/section[1]/span[2]/button""")
    Comment.click()


def SendComment():
    SendComment = driver.find_element(By.XPATH,
                                      """/html/body/div[2]/div/div/div[3]/div/div/div[1]/div/div[3]/div/div/div/div/div[2]/div/article/div/div[2]/div/div/div[2]/section[3]/div/form/div/textarea""")
    SendComment.send_keys(random.choice(ListofComments))
    SendComment.send_keys(Keys.ENTER)


def ChanceofComment():
    random_number = random.random()
    if random_number < 0.34:
        time.sleep(2)
        Comment()
        time.sleep(5)
        SendComment()
        print("Commented on Post")
        print('\n')


def FirstImageNext():
    Next = driver.find_element(By.XPATH,
                               """/html/body/div[2]/div/div/div[3]/div/div/div[1]/div/div[3]/div/div/div/div/div[1]/div/div/div/button""")
    Next.click()


def MainBot():

    time.sleep(4)
    try:
        ImageLike()
        print("Post Liked")
        print('\n')
    except Exception as e:
        print("Error when liking image, will skip post")
        print('\n')
        NextPhoto()
        MainBot()

    time.sleep(4)
    try:
        Follow()
        print("User Followed")
        print('\n')
    except Exception as e:
        print("Error when following user, will skip photo")
        print('\n')
        NextPhoto()
        MainBot()

    time.sleep(4)
    try:
        ChanceofComment()
    except Exception as e:
        print("Error when commenting, will skip photo")
        print('\n')
        NextPhoto()
        MainBot()

    time.sleep(4)
    time.sleep(4)
    NextPhoto()
    global Counter
    Counter = Counter + 1


def FirstImage():
    Image = WebDriverWait(driver,
                          20).until(EC.element_to_be_clickable((By.CLASS_NAME,
                                                                '_aagw')))
    time.sleep(2)
    try:
        Image.click()
        print("First Image Clicked")
        print('\n')
    except Exception as e:
        print("Error first image in hastag could not be pressed, refreshing...")
        print('\n')
        time.sleep(2)
        driver.get(instagramUrlTag + HashTag)
        time.sleep(5)
        FirstImage()

    time.sleep(4)
    try:
        ImageLike()
        print("Post Liked")
        print('\n')
    except Exception as e:
        print("Error when liking post, will skip post...")
        print('\n')
        time.sleep(2)
        FirstImageNext()
        time.sleep(2)
        LikeAndCommentCycle(Likes)
    time.sleep(4)
    try:
        Follow()
        print("User Followed")
        print('\n')
    except Exception as e:
        print("Error on following user, will skip post...")
        print('\n')
        time.sleep(2)
        FirstImageNext()
        time.sleep(2)
        LikeAndCommentCycle(Likes)
    time.sleep(4)
    try:
        ChanceofComment()
    except Exception as e:
        print("Error when commenting, will skip post...")
        print('\n')
        time.sleep(2)
        FirstImageNext()
        time.sleep(2)
        LikeAndCommentCycle(Likes)
    time.sleep(2)

    time.sleep(4)

    # Unique for the first image
    FirstImageNext()
    global Counter
    Counter = Counter + 1
    if Likes == 1:
        print("Just one photo...Finished")
        print('\n')
        quit()

    LikeAndCommentCycle(Likes)


FirstImage()
print("Finished! Interacted with {} Posts".format(Likes))
print('\n')
time.sleep(10)
driver.quit()
