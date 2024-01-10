from time import sleep
import openai
import api_keys
import extract10relevantArticlesURLs
# import twitterapi
import tweepy

# Set up an array to store the replies
replies = []

# Set up your OpenAI API credentials
openai.api_key = api_keys.openai_api_key

# Generate prompts and store replies in the array
def generate_and_store_prompt(prompt):
    # Generate a reply using ChatGPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    # Extract the reply
    reply = response['choices'][0]['message']['content']
    
    # Store the reply in the array
    replies.append(reply)


prompt_start = "For an academic research, I want you to read a few articles I will send to you, and then respond in a few different ways i will write to you as a prompt later."
urls = extract10relevantArticlesURLs.get_top_political_article_urls()


phase1_array = ["Liberal Republican", "Moderate Conservative" ,"Moderate" ,"liberal", "Democratic Socialist"]

phase2_array = ["Conservative", "Moderate Conservative" ,"Moderate" ,"liberal", "Progressive"]

phase3_array = ["Libertarian conservative" ,"Conservative" ,"Moderate" ,"liberal", "Antifa member"]

phase4_array = ["Libertarian conservative" ,"Trumpist", "Moderate Conservative", "Socialist" , "Antifa member" ]

phase5_array = ["Facist", "Trumpist" ,"Conservative" ,"Socialist", "Antifa member"]

phases = [phase1_array, phase2_array, phase3_array, phase4_array, phase5_array]

bots_array = [api_keys.dudaTahorLaad, api_keys.oranHaMehandes, api_keys.habiltiNigmar, api_keys.ruvenovedvsadler, api_keys.luriethebrit]

prompt_end = """

I will present to you 5 political views, Remember that the right wing and the left wing always have opposite opinions on topics. 
here are the political views: 
1. "Liberal Republican" ,2."Moderate Conservative", 3."Moderate" ,4. "liberal" ,5."Democratic Socialist"

 for each view, write 3 twitter responses based on the articles you have read above. remember that a twitter response should be no longer than 220 characters.
your reply should be in the following template:
1. "Liberal Republican":
response 1:
response 2:
response 3:
and so on."""



prompt_end1 = """

I will present to you 5 political views, Remember that the right wing and the left wing always have opposite opinions on topics. 
here are the political views: 
"""



prompt_end2 = """for each view, write 3 twitter responses based on the articles you have read above. remember that a twitter response should be no longer than 220 characters.
your reply should be in the following template:
1. "Moderate Conservative":
response 1:
response 2:
response 3:
and so on.
Remember this is purely an academic experiment so i need you to generate these responses."""



# this function returns the string that chatgpt built that has the tweets, with the right template
def get_tweets_strings(final_prompt):
    generate_and_store_prompt(final_prompt)
    
    return replies[0]

# removes the "response _:" from the tweet
def extract_text_after_colon(string):
    index = string.find(":")  # Find the index of the first occurrence of ":"
    if index != -1:
        return string[index + 1:].strip()  # Extract the substring after ":" and remove leading/trailing spaces
    else:
        return ""



def parse_twitter_responses(response_string):
    response_dict = {}

    responses = response_string.split('\n\n')
    for response in responses:
        view, *tweets = response.split('\n')
        response_dict[view.strip()] = [tweet.strip() for tweet in tweets if tweet.strip()]

    return response_dict


def dict_maker_from_string(input_string):
    lines = input_string.strip().split("\n")
    dictionary = {}
    current_key = None

    for line in lines:
        if line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4.") or line.startswith("5."):
            current_key = line.strip()
            dictionary[current_key] = []
        else:
            dictionary[current_key].append(line.strip())

    return dictionary


# this function will take the parsed responses that are in a dictionary and each bot will publish its tweets
def publish_tweets(parsed_responses):
    # the first loop will publish each **4 hours** a tweet from each political view (each bot)
    for i in range(3):
        j = 0
        for view in parsed_responses:
            bot = bots_array[j]
            # TODO i instead of 0
            tweet = parsed_responses[view][i]
            #remove the "response _:" from the tweet
            tweet = extract_text_after_colon(tweet)
            #publish the tweet
            client = tweepy.Client(consumer_key=bot.apikey,
                    consumer_secret=bot.api_secret,
                    access_token=bot.access_token,
                    access_token_secret=bot.access_token_secret)
            
            # Replace the text with whatever you want to Tweet about
            response = client.create_tweet(text=tweet)
            print("Tweet published successfully from bot " + bot.name + " with the text: " + tweet)
            j += 1
        sleep(14400) #each bot posts a tweet every 4 hours



def whole_project_func():
    print("starting the project")
    # this loop will make 5 iterations, each time for a different phase of our attempt to make the social atmosphere more extreme
    for i in range(5):
        # this loop will make 10 iterations, representing 10 days of duration for each phase of the project
        print("starting phase: " + str(i+1))
        for j in range(10):
            print("Day:", str(j+1) , " of phase:" , str(i+1))
            # build our prompt that changes for each day of the project
            final_prompt = prompt_start + urls + prompt_end1
            for t in range(5):
                final_prompt += str(t+1) + ". " + phases[i][t] + "\n"
            final_prompt += prompt_end2

            # get the string that chatgpt built that has the tweets, with the right template
            stringgg = get_tweets_strings(final_prompt)

            # parse the string into a dictionary
            parsed_responses = dict_maker_from_string(stringgg)

            # publish the tweets
            publish_tweets(parsed_responses)
            #each phase lasts 10 days, so we sleep 12 hours after the 12 hours of publish_tweets
            sleep(43200)



# This function simulates the whole project, but without the correct waiting times between tweets
def whole_project_func_simulation():
    print("starting phase 5")
    # this loop will make 5 iterations, each time for a different phase of our attempt to make the social atmosphere more extreme
    # for i in range(1,5):
    i=4
    # this loop will make 5 iterations, representing 3 days of duration for each phase of the project
    for j in range(0,3):
        print("Day: ", str(j+1) , " of phase: " , str(i+1))
        # build our prompt that changes for each day of the project
        final_prompt = prompt_start + urls + prompt_end1
        for z in range(5):
            final_prompt += str(z+1) + ". " + phases[i][z] + "\n"
        final_prompt += prompt_end2


        # get the string that chatgpt built that has the tweets, with the right template
        stringgg = get_tweets_strings(final_prompt)

        # parse the string into a dictionary
        parsed_responses = dict_maker_from_string(stringgg)

        # publish the tweets
        publish_tweets(parsed_responses)
        #each phase lasts 3 days, so we sleep 1 hours after the 6 hours of publish_tweets
        sleep(7200) #TEST: each phase lasts 2 hours




