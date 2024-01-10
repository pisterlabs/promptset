import random
import os
import time
import string
import openai
import dotenv
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

# download the vader lexicon for sentiment analysis
# nltk.download('vader_lexicon') # explain vader lexicon : https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/
# Ignore warnings from sklearn and nltk libraries for cleaner output in the terminal
warnings.filterwarnings(action='ignore', category=UserWarning) 

# Load the .env file
dotenv.load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the Sim class it's attributes and methods
class Sim:
    # Define the STATS for the Sim class
    def __init__(self):
        self.needs = {
            "hunger": 10,
            "hygiene": 10,
            "bladder": 10,
            "energy": 10,
            "social": 10,
            "fun": 10,
            "environment": 10,
            "comfort": 10
        }
        self.mood = "happy"  # Add mood attribute
        self.days = 0  # Add days attribute
        self.SimsJournal = {}  # Add SimsJournal attribute

    # reduce needs method to reduce the needs of the Sim class by a random number between 1 and 3
    def reduce_needs(self):
        for need in self.needs:
            self.needs[need] = max(0, self.needs[need] - random.randint(1, 3))
            if self.needs[need] == 0:
                return need
        return None
    
    # calculate mood method to calculate the mood of the Sim class
    def calculate_mood(self):
        total_needs = sum(self.needs.values())
        if total_needs < 30:
            self.mood = "stressed"
        elif total_needs >= 30 and total_needs < 60:
            self.mood = "satisfied"
        else:
            self.mood = "happy"

    # checks needs for lowest need method to check the needs of the Sim class and return the lowest need
    def check_needs(self):
        lowest_need = min(self.needs, key=self.needs.get)
        return lowest_need
    
    # print needs method to print the needs of the Sim class
    def print_needs(self):
        print(f"Mood: {self.mood.capitalize()}")
        for need, value in self.needs.items():
            print(f"{need.capitalize()}: {value}")


    # Sims logic for choosing an item to use
    def choose_item(self):
        sorted_needs = sorted(self.needs.items(), key=lambda x: x[1])
        for need, value in sorted_needs:
            if value < 3:
                return need
        return None

# Define the simsRoom class it's attributes and methods, this is the room the Sim is in
# It includes all the items that advertise their ability to fill a Sims need
# It also includes the journal logic to keep track of the Sims activities and mood
class simsRoom:
    def __init__(self):
        self.theSim = Sim()
        self.needs_to_items = {
            "hunger": self.use_fridge,
            "hygiene": self.use_shower,
            "bladder": self.use_toilet,
            "energy": self.use_bed,
            "fun": self.use_tv,
            "environment": self.use_painting_on_the_wall,
            "comfort": self.use_couch,
            "social": self.use_telephone
        }

    def use_fridge(self):
        print("\nThe Sim is using the fridge to grab something to eat.")
        self.theSim.needs["hunger"] = min(10, self.theSim.needs["hunger"] + 8)
        self.theSim.needs["energy"] = max(0, self.theSim.needs["energy"] - 1)  # Deduct 1 energy
        self.update_journal("used fridge")  # Update journal
        # add wait time
        time.sleep(1)

    def use_shower(self):
        print("\nThe Sim is having a shower.")
        self.theSim.needs["hygiene"] = min(10, self.theSim.needs["hygiene"] + 8)
        self.theSim.needs["comfort"] = min(10, self.theSim.needs["comfort"] + 2)
        self.theSim.needs["environment"] = min(10, self.theSim.needs["environment"] + 1)
        self.theSim.needs["energy"] = max(0, self.theSim.needs["energy"] - 1)  # Deduct 1 energy
        self.update_journal("took a shower")  # Update journal
        # add wait time
        time.sleep(1)

    def use_toilet(self):
        print("\nThe Sim is using the toilet.")
        self.theSim.needs["bladder"] = min(10, self.theSim.needs["bladder"] + 4)
        self.theSim.needs["energy"] = max(0, self.theSim.needs["energy"] - 1)  # Deduct 1 energy
        self.update_journal("used toilet")  # Update journal
        # add wait time
        time.sleep(1)

    def use_bed(self):
        print("\nThe Sim is going to sleep.")
        self.theSim.needs["energy"] = min(10, self.theSim.needs["energy"] + 10)
        self.theSim.needs["comfort"] = min(10, self.theSim.needs["comfort"] + 10)
        self.update_journal("went to sleep")  # Update journal
        self.theSim.days += 1  # Increment days
        self.theSim.SimsJournal[self.theSim.days] = []  # Start a new day in the journal
        # add wait time
        time.sleep(1)

    def update_journal(self, activity):
        if self.theSim.days not in self.theSim.SimsJournal:
            self.theSim.SimsJournal[self.theSim.days] = []  # Initialize a new day in the journal
        self.theSim.SimsJournal[self.theSim.days].append({"activity": activity, "mood": self.theSim.mood})  # Update SimsJournal
        

        # Save the journal to a JSON file
        with open('sims_journal.json', 'w') as f:
            json.dump(self.theSim.SimsJournal, f)

    def use_tv(self):
        print("\nThe Sim is watching TV.")
        self.theSim.needs["fun"] = min(10, self.theSim.needs["fun"] + 3)
        self.theSim.needs["comfort"] = min(10, self.theSim.needs["comfort"] + 2)
        self.theSim.needs["environment"] = min(10, self.theSim.needs["environment"] + 1)
        self.theSim.needs["energy"] = max(0, self.theSim.needs["energy"] - 1)  # Deduct 1 energy
        self.update_journal("watched TV")  # Update journal
        # add wait time
        time.sleep(1)

    def use_couch(self):
        print("\nThe Sim is sitting on the couch.")
        self.theSim.needs["comfort"] = min(10, self.theSim.needs["comfort"] + 8)
        self.theSim.needs["fun"] = min(10, self.theSim.needs["fun"] + 1)
        self.theSim.needs["energy"] = max(0, self.theSim.needs["energy"] + 1)
        self.update_journal("sat on the couch")  # Update journal
        # add wait time
        time.sleep(1)

    def use_painting_on_the_wall(self):
        print("\nThe Sim is looking at the framed painting.")
        self.theSim.needs["environment"] = min(10, self.theSim.needs["environment"] + 5)
        self.theSim.needs["fun"] = min(10, self.theSim.needs["fun"] + 5)
        self.theSim.needs["energy"] = max(0, self.theSim.needs["energy"] - 1)  # Deduct 1 energy
        self.update_journal("looked at the framed painting")  # Update journal
        # add wait time
        time.sleep(1)

    def use_telephone(self):
        print("\nThe Sim is making a phonecall.")
        self.theSim.needs["social"] = min(10, self.theSim.needs["social"] + 5)
        self.theSim.needs["fun"] = min(10, self.theSim.needs["fun"] + 2)
        self.theSim.needs["energy"] = max(0, self.theSim.needs["energy"] - 1)
        self.update_journal("talking on the phone")  # Update journal
        telephone_call = TelephoneCall(self.theSim)
        telephone_call.start_call()

    # This is the turns manager, it will run the game until the user types 'quit'
    # It also checks the needs of the Sim and calls the appropriate methods
    # It also calls the choose_item method to choose an item for the Sim to use
    # It also calls the reduce_needs method to reduce the needs of the Sim
    
    def turns_manager(self):
        #print("Type 'quit' to end the chat.")
        while True:
            need = self.theSim.check_needs()
            if need:
                self.needs_to_items[need]()
                print("\nSim Stats:")
                self.theSim.print_needs()
                self.theSim.calculate_mood()  # Update the Sim's mood after each need is updated
                print()

            user_input = input("\nPress Enter to proceed or type 'quit' to end the chat: ")
            if user_input.lower() == "quit":
                break

            need = self.theSim.reduce_needs()
            if need:
                self.needs_to_items[need]()
                print("\nSim Stats:")
                self.theSim.print_needs()
                self.theSim.calculate_mood()  # Update the Sim's mood after each need is updated
                print()

            chosen_item = self.theSim.choose_item()
            if chosen_item:
                if self.theSim.needs[chosen_item] > 8:
                    #print(f"The Sim's {chosen_item} need is too high. Rechoosing item...")
                    chosen_item = self.theSim.choose_item()
                self.needs_to_items[chosen_item]()
                print("\nSim Stats:")
                self.theSim.print_needs()
                self.theSim.calculate_mood()  # Update the Sim's mood after each need is updated
                print()
                if self.theSim.needs[chosen_item] < 1:
                    print(f"The Sim needs to address {chosen_item}. Sim hangs up.")
                    break


# This is the telephone call logic it will run until the user types 'hangup' or the sims needs are too low, if a need reaches 0 the call will end.
# It also calls the get_sentiment method to get the sentiment of the user input, and the get_keywords method to get the keywords of the user input
# It also handles json file reading and writing
class TelephoneCall:
    def __init__(self, sim):
        self.sim = sim
        self.conversation = []

    def save_conversation(self):
        try:
            with open('conversation.json', 'r') as f:
                existing_conversations = json.load(f)
        except FileNotFoundError:
            existing_conversations = []

        existing_conversations.append(self.conversation)

        with open('conversation.json', 'w') as f:
            json.dump(existing_conversations, f, indent=4)  # Save updated conversation dictionary to JSON file


    def start_call(self):
        print("Telephone is ringing...")
        print("Sim: Hello?")
        self.conversation.append({"role": "Sim", "content": "Hello?", "sentiment": None, "keywords": []})  # Add empty sentiment and keywords list to each message
        while True:
            user_input = input("User: ")
            user_sentiment = get_sentiment(user_input)  # Get the sentiment of the user input
            self.conversation.append({"role": "User", "content": user_input, "sentiment": user_sentiment, "keywords": []})  # Add the user input and its sentiment to the conversation

            # Format the prompt in a more conversational way
            needs_str = ', '.join(f'{k} is at {v}' for k, v in self.sim.needs.items())
            prompt = f"Sim: My current needs are: {needs_str}.\nUser: {user_input}\n"
            response = openai_chat(prompt, self.sim.mood, self.sim.SimsJournal)
            sim_sentiment = response[1]  # Get the sentiment of the Sim's response
            print(f"Sim: {response[0]}")  # Print the response to the console
            self.conversation.append({"role": "Sim", "content": response[0], "sentiment": sim_sentiment, "keywords": []})  # Store both the response content and sentiment in the conversation
            self.extract_keywords()  # Extract keywords after each message is added
            need = self.sim.reduce_needs()
            if need is not None:
                print(f"The Sim needs to address {need}. They hang up abruptly.")
                self.save_conversation()
                return
            self.sim.calculate_mood()  # Calculate the Sim's mood based on their total needs score
            if user_input.lower() == "hangup":
                print("You hang up the phone.")
                
                self.save_conversation()
                return



    def extract_keywords(self):
        vectorizer = TfidfVectorizer(tokenizer=lambda text: [word for word in word_tokenize(text) if len(word) > 5], stop_words=stopwords.words('english'))
        # Convert list of dictionaries to list of messages for TF-IDF transformation
        messages = [message["content"] for message in self.conversation]
        tfidf_matrix = vectorizer.fit_transform(messages)
        feature_names = vectorizer.get_feature_names_out()
        for i in range(len(messages)):
            tfidf_scores = dict(zip(feature_names, tfidf_matrix[i].toarray().flatten()))
            sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
            top_keywords = sorted_keywords[:10]  # Get top 10 keywords
            # Remove punctuation from keywords
            top_keywords = [keyword for keyword in top_keywords if keyword[0] not in string.punctuation]
            if top_keywords:
                self.conversation[i]["keywords"] = top_keywords[0][0]  # Save top keyword in conversation dictionary
            else:
                self.conversation[i]["keywords"] = None

# This is  the sentiment analysis for the user input, it is different from the sentiment analysis for the sims response.
def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

    

def openai_chat(prompt, mood, journal):
        # Set temperature based on mood
    if mood == "happy":
        temperature = 0.8
    elif mood == "satisfied":
        temperature = 0.6
    elif mood == "stressed":
        temperature = 0.4
    else:
        temperature = 0.5  # Default temperature

    current_day = max(journal.keys())
    current_day_activities = ', '.join(f'{entry["activity"]} (mood: {entry["mood"]})' for entry in journal[current_day])  # Get all activities for the current day
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=temperature,
        max_tokens=250,
        messages=[
            {"role": "system", "content": f"You are a Sim from The Sims. You are chatting with a user on the phone. Your room has a fridge, a shower, a toilet, a painting, a couch and a tv. You are currently feeling {mood}. Here is what you did today: {current_day_activities}"},  # Add current day activities to system message
            {"role": "user", "content": prompt},
        ],
    )
    response_content = response['choices'][0]['message']['content'].strip()
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment = sentiment_analyzer.polarity_scores(response_content)
    return response_content, sentiment

def start_simulation():

    print("Welcome to Teh Sims")
    input("Press Enter to start...")
    chat_room = simsRoom()
    chat_room.turns_manager()

start_simulation()    

