# TODO: This script requires unit tests and should be crash-tested 
#       (with special focus on boundaries, edge and corner scenarios) to ensure solid error handling. 
#       Stress testing would further enable us to optimize its performance

import openai
import os
import atexit
import sqlite3
from collections import Counter
import random
import time
from textblob import TextBlob
from threading import Thread
import transitions
from transitions.extensions.states import Timeout, Tags, add_state_features
from transitions.extensions.diagrams import GraphMachine
from itertools import cycle
from typing import Dict, List
import vaderSentiment.vaderSentiment as vader
import requests
from bs4 import BeautifulSoup

from github import Github
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
OWNER = "iterabloom"
REPO_NAME = "talkasaurus.rex.flask"
github = Github(GITHUB_PERSONAL_ACCESS_TOKEN)
repo = github.get_repo(f"{OWNER}/{REPO_NAME}")


class DevOpsBot:
    """
    encloses various utilities geared towards Github issue tracking and deployment facilitation. 
    This rudimentary bot lays down the groundwork but is presently mostly limited to managing Github 
    issues and prompting manual_review for complicated situations.
    """
    def __init__(self):
        self.github = Github(GITHUB_PERSONAL_ACCESS_TOKEN)
        self.repo = self.github.get_repo(f"{OWNER}/{REPO_NAME}")
        self.codebase = self.fetch_codebase()

    def fetch_codebase(self):
        contents = self.repo.get_contents("")
        code = {}
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(self.repo.get_contents(file_content.path))
            else:
                code[file_content.name] = file_content.decoded_content.decode()
        return code

    def generate_code(self, feature_desc: str, language: str) -> str:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"As a developer fluent in {language}, write functional code for the following feature: {feature_desc}",
            temperature=0.5,
            max_tokens=200,)
        return response.choices[0].text.strip()

    def review_code(self, code_snippet: str, language: str) -> List[str]:
        prompt = f"As an experienced developer, review the following {language} code and suggest improvements: {code_snippet}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=200,
        )
        if "[manual_review]" in response.choices[0].text:
            self.manual_review(response.choices[0].text, )
        return response.choices[0].text.strip().split("\n")

    def manual_review(self, review):
        self.repo.create_issue(title="[MANUAL REVIEW NEEDED] Code Snippet Alert", body=f"Manual review is needed for the following code: {review}")

    def new_feature(self, feature_desc: str, language: str):
        new_code = self.generate_code(feature_desc, language)
        review = self.review_code(new_code, language)
        return new_code, review

    def watch_issues(self):
        issues_at_hand = self.repo.get_issues(state='open')
        other_issues = []
        for issue in issues_at_hand:
            if "[MANUAL REVIEW NEEDED]" in issue.title:
                # This is a pseudo function to notify a developer on the project to check this part of the code for further issues or problems and to manually write a test for this.
                self.notify_developer(issue)
            else:
                other_issues.append(other_issues)
        return other_issues


@add_state_features(Timeout, Tags)
class CustomStateMachine(GraphMachine):
    pass


class UserConvoFlags:
    # Enumeral class to get flag against the detected user's conversation events
    NO_EVENT = 0
    INTERRUPTION = 1
    INTERJECTION = 2
    TIME_GAP = 3
    HISTORY_RECALL = 4
    #...


class ConversationEvents:
    # Class object to encapsulate the occurrence of the events in the conversation
    def __init__(self):
        self.event = UserConvoFlags.NO_EVENT
        self.event_cycle = cycle(UserConvoFlags)
        self.set_conversation_event()

    def set_conversation_event(self):
        self.event = next(self.event_cycle)


class ConversationHandler:
    """
    Class for handling conversation sequencing, 
    feedback, overlap, interruption, and repairing mistakes in conversation

    This class contains a CustomStateMachine object self.machine configured with the appropriate states 
    (e.g. 'initial', 'normal', 'interrupted') and transitions (Triggers and resulting states e.g. 'interrupt'). 
    The 'machine' follows the designated transition depending on the trigger from user conversation event.

    There is a continuous multithread job '_task_listener' for identifying user conversation events and triggering 
    state transition corresponding to the event. A UserConvoFlags has been used as enumeration for representing different 
    conversation states. It is in the '_task_listener' where each actual mapping of the 
    conversation event to the state transition is done.

    Please note that the implementation of methods like 'backup_context', 'conversation_timeout', 
    'reply_to_interjection' etc. that are being used in providing behavior to the conversation state machine 
    are not outlined and would need to be written. Their name outlines the functionality they should have, e.g., 
    backup_context method would hold the conversation context in face of an event ('interrupt' in this case) 
    which might disrupt ongoing conversation. 
    """

    states = ['initial', 'normal', 'interrupted', 'interjection', 'long_pause', 
              'history_recall', 'contextual_response', 'non_contextual_response']

    transitions = [
        {'trigger': 'interrupt', 'source': '*', 'dest': 'interrupted', 'before': 'backup_context',
         'after': 'conversation_timeout'},
        {'trigger': 'interject', 'source': '*', 'dest': 'interjection', 'after': 'reply_to_interjection'},
        {'trigger': 'pause', 'source': 'normal', 'dest': 'long_pause', 'after': 'reset_pipeline'},
        {'trigger': 'history_recall', 'source': '*', 'dest': 'history_recall', 'after': 'recall_history'},
        {'trigger': 'provide_information', 'source': '*', 'dest': 'contextual_response', 'after': 'provide_backchannel'},
        {'trigger': 'respond_with_fallback', 'source': '*', 'dest': 'non_contextual_response'},
        {'trigger': 'resume', 'source': '*', 'dest': 'normal'},
    ]

    def state_timer(self):
        #TODO: setup a timer to transition back to 'normal' state after a set while.
        #      This allows handling of unexpected lags or delays in user responses.
        pass

    def __init__(self, buffer_size=10):
        self.message_history = []
        self.buffer_size = buffer_size
        self.conn = sqlite3.connect('messages.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, message TEXT)')
        atexit.register(self._cleanup)
        self.user_convo_events = ConversationEvents()  #TODO: check whether this simulates user events in an unending cycle
        self.machine = CustomStateMachine(model=self, states=ConversationHandler.states, transitions=ConversationHandler.transitions, initial='initial')
        self.conversation_task_auto()

    def process(self, message):
        #TODO: update to include generation of conversation events from user messages
        self.cursor.execute("INSERT INTO messages (message) VALUES (?)", (message,))
        self.conn.commit()

        if len(self.message_history) >= self.buffer_size:  # control buffer size
            self.message_history.pop(0)

        self.message_history.append(message)
        return message

    def generate_prompts(self, user_tone, conversation_state):
        # Based on the user's tone and the current conversation state, generate a list of prompts
        prompts = []
        # TODO: Add logic for generating prompts
        return prompts

    def _cleanup(self):
        self.conn.close()
    
    def conversation_task_auto(self):
        Thread(target=self._task_listener).start()

    def _task_listener(self):
        while True:
            if self.user_convo_events.event == UserConvoFlags.INTERRUPTION:
                self.interrupt()
            elif self.user_convo_events.event == UserConvoFlags.INTERJECTION:
                self.interject()
            elif self.user_convo_events.event == UserConvoFlags.TIME_GAP:
                self.pause()
            elif self.user_convo_events.event == UserConvoFlags.HISTORY_RECALL:
                self.history_recall()
            else:
                # handle other events
                pass
            self.user_convo_events.set_conversation_event()
            time.sleep(1)


class UserAdaptability:
    """
    Class to understand user's lexicon, syntax, overtones, and conversational patterns,
    as well as retain long-term explicit instructions, implement personalized interests and research,
    and utilize sentiment analysis.

    analyze_sentiment() uses the vaderSentiment library to analyze sentiment in a message, 
    and fetch_research_info() uses web scraping to fetch information on a topic from Wikipedia.

    act_on_instructions() checks if there are any outstanding tasks. It flags them as completed once 
    the associated action has been performed (in this case, research on a specific topic).

    The actual implementation for more complex tasks requires additional planning and substantial investment. 
    However, the functionalities described above provide a foundation that can readily be expanded on. 
    This revised class achieves the goal of making TalkasaurusRex adaptable to different user preferences.
    """

    def __init__(self):
        self.user_lexicon = set()
        self.user_mannerisms = Counter()
        self.user_sentiments = []
        self.pos_tags = Counter()
        self.user_tasks = []
        self.user_instructions = []
        self.long_term_instructions = []
        self.conn = sqlite3.connect('user_adaptability.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS lexicon (id INTEGER PRIMARY KEY, word TEXT)""")
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS mannerisms (id INTEGER PRIMARY KEY, mannerism TEXT,
                            frequency INTEGER)""")
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS sentiments (id INTEGER PRIMARY KEY, sentiment REAL)""")
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS pos_tags (id INTEGER PRIMARY KEY,
                            pos_tag TEXT, frequency INTEGER)""")
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS user_tasks (task_id INTEGER PRIMARY KEY, task TEXT,
                              is_complete INTEGER DEFAULT 0)""")
        atexit.register(self._cleanup)

        # Initialize sentiment analyzer
        self.sentiment_analyzer = vader.SentimentIntensityAnalyzer()

    def adapt(self, message):
        self.user_lexicon.update(set(word.lower() for word in message.split(' ')))

        for word in self.user_lexicon:
            self.cursor.execute("INSERT INTO lexicon (word) VALUES (?)", (word,))

        blob = TextBlob(message)
        self.user_mannerisms.update(blob.word_counts)
        self.user_sentiments.append(blob.sentiment.polarity)
        self.pos_tags.update(tag for (word, tag) in blob.tags)

        for mannerism, frequency in self.user_mannerisms.items():
            self.cursor.execute('INSERT INTO mannerisms (mannerism, frequency) VALUES (?, ?)', (mannerism, frequency))

        for sentiment in self.user_sentiments:
            self.cursor.execute("INSERT INTO sentiments (sentiment) VALUES (?)", (sentiment,))

        for pos_tag, frequency in self.pos_tags.items():
            self.cursor.execute('INSERT INTO pos_tags (pos_tag, frequency) VALUES (?, ?)', (pos_tag, frequency))
        
        user_instruction = self.detect_explicit_instruction(message)
        if user_instruction:
            self.cursor.execute("INSERT INTO user_tasks (task) VALUES (?)", (user_instruction,))
            self.user_instructions.append(user_instruction)

        self.conn.commit()

        return message

    def _cleanup(self):
        self.conn.close()

    def detect_explicit_instruction(self, message):
        # TODO: This is a very basic example and may be substituted with an API call with GPT 4 or other ML models.
        #       In this example, I just use the word "please" as an instruction indicator
        instruction = None
        if "please" in message.lower():
            instruction = message
        return instruction

    def analyze_sentiment(self, message):
        """
        Analyze the sentiment of a message using VADER Sentiment Analysis.
        Returns a compound score which is a computed metric that sums the intensities
        of each word in the lexicon, adjusted according to the rules, and then normalized
        to be between -1 (most extreme negative) and +1 (most extreme positive).
        """
        return self.sentiment_analyzer.polarity_scores(message)['compound']

    def fetch_research_info(self, topic):
        """
        Fetches information from internet to conduct research on designated topics.
        It is a placeholder and can be modified according to complexity of tasks. Better to use an extarnal API service.
        """
        URL = f"https://en.wikipedia.org/wiki/{topic}"
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")

        paragraphs = soup.select("p")
        research_info = " ".join([para.text for para in paragraphs[:3]])

        return research_info

    def act_on_instructions(self):
        self.cursor.execute("SELECT * FROM user_tasks")
        rows = self.cursor.fetchall()
        for row in rows:
            task_id, task, is_complete = row
            if not is_complete and "research" in task:
                self.cursor.execute('UPDATE user_tasks SET is_complete = 1 WHERE task_id = ?', (task_id,))
                research_topic = task.split(" ")[-1]  # assume the research topic comes after "research"
                info = self.fetch_research_info(research_topic)
                return info
        return None


class PromptEngineer:
    """
    Generates schema of prompts to understand different aspects of user's speech. 

    By simulating various expert roles, these prompts allow the system to analyze the user's speech 
    in a diverse and comprehensive manner. The outputs can then be used to provide custom feedback to the user, 
    shaping the behavior of the conversational AI based on the user's preferences. 

    One potential limitation of this approach is that it may exceed your OpenAI API usage, particularly for long 
    or frequent conversations, as it multiples API usage by the number of viewpoints included. 
    However, for an intensive analysis of the user's conversation style (the premise of PromptEngineer), 
    it would be a reasonable tradeoff. 
    """

    def __init__(self, adaptability_module):
        self.adaptability_module = adaptability_module

    def linguist_view(self):
        """
        Focuses on the structure of language and syntax.
        """
        conversation_prompts = [
            {"role": "system", "content": "You are an expert in Linguistics."},
            {"role": "user", "content": "Analyze the syntax and structure of my speech."},
            {"role": "assistant", "content": "The user often uses the following words and phrases: {0}.".format(', '.join(self.adaptability_module.user_lexicon))}
        ]
        return self._pass_to_gpt4(conversation_prompts)

    def psycholinguist_view(self):
        """
        Considers psychological factors in understanding language.
        """
        avg_sentiment = sum(self.adaptability_module.user_sentiments) / len(self.adaptability_module.user_sentiments) if self.adaptability_module.user_sentiments else 0
        sentiment_type = "positive" if avg_sentiment > 0 else "negative" if avg_sentiment < 0 else "neutral"
        conversation_prompts = [
            {"role": "system", "content": "You are an experienced Psycholinguist."},
            {"role": "user", "content": 'What does my language reveal about my mindset?'},
            {"role": "assistant", "content": f"The user usually uses {sentiment_type} language."}
        ]
        return self._pass_to_gpt4(conversation_prompts)

    # Implement other views (sociolinguist_view, phonetic_view, etc.) in the same fashion here...

    def _pass_to_gpt4(self, conversation_prompts):
        """
        TODO: maybe chat_with_gpt_model can be imported from api_server.py?
        Functionality for prompt-engineering via the GPT-4 API. You'll need to replace 'gpt-4.0-turbo' with your model name.
        """
        conversation = chat_with_gpt_model(conversation_prompts)
        result = conversation['choices'][0]['message']['content'] if conversation else "I'm sorry, I didn't understand that."
        return result




def api_delay_error_handler(api_method, retries=5):
    """
    TODO: update to except onResponse's status code
    Retry API method in the event of failure up to indicated retries with a delay.
    """
    attempts = 0
    while attempts < retries:
        try:
            result = api_method()
            break
        except Exception as e:
            print(e)
            if attempts < retries-1:  
                wait_time = (2 ** attempts) + (random.randint(0, 1000) / 1000)
                print(f"API call failed. Retrying in {wait_time} seconds.")
                time.sleep(wait_time)
                attempts += 1
            else: 
                print("API call failed after several attempts.")
                result = None
    return result


# TODO: I think this is duplicative of a function in api_server.py
GPT_MODEL = "gpt-4-32k"
def chat_with_gpt_model(model=GPT_MODEL, api_key=MY_API_KEY, messages=[], max_tokens=150):
    api_call = lambda: openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    return api_delay_error_handler(api_call)