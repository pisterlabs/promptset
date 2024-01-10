from linkedin_api import Linkedin
import openai
from dotenv import load_dotenv
import os

# # change the input to your own data/login
# api = Linkedin("theEmailYouUsedToLoginForLinkedIn@gmail.com","thePasswordAssociatedWithThatAccount")
# data = api.get_profile("enterUsernameHereToQueryData")
#
# # get user's summary
# summary = data["summary"]


# Define the prompt
# prompt = "A good linkedin profile summary includes a greeting, one's professional interest or current focus, one's personal interests (optional), " \
#          "and one's contact information. \nA good example of this is: \n" \
#          "(greeting) Thanks for taking the time to get to know me through my Linkedin profile. I appreciate you stopping by!\n"\
#          "(professional interest or current focus) Professionally, I am passionate about creating and delivering meaningful learning experiences that help sales professionals use LinkedIn Sales Navigator to target, understand, and engage their prospects and customers.\n"\
#          "(personal interests (optional)) Personally, I'm a high energy person with a passion for building relationships through genuine interests in others. I am a proud father to a daughter and son, who keep my wife and I busy, while bringing us entertainment and joy. I enjoy spending time outdoors (golfing, running), weightlifting, NFL football (Go Birdz!), and listening to great music.\n" \
#          "(contact information) Feel free to send me a personalized connection request or contact me directly at ...email.com if you wish to discuss mutual areas of interest or to learn more.\n" \
#          "Rate the following prompt on a scale of 1 to 5, where 1 represents unsatisfactory and 5 is exemplary from the above analysis and include an analysis of what the prompt should include and why: \n"
#
# print(prompt + summary)

def getGPT():
    load_dotenv()  # take environment variables from .env.

    openai.api_key = os.getenv("OPENAI_API_KEY")
    completion = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Say this is a test",
        max_tokens=50,
        temperature=0
    )
    # Print the generated text
    generated_text = completion.choices[0].text
    #print(generated_text)

    return generated_text