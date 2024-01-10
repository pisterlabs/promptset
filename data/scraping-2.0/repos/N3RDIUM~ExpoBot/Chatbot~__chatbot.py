import random
import json
import inflect
import fuzzywuzzy.fuzz as fuzz
import spacy
import _sha256 as sha256
import tqdm
import yaml
import os

p = inflect.engine()
import logging
logging.basicConfig(level=logging.INFO)

THRESHOLD = 2 # Similarity threshold

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import openai
import json
# if not "USELOCAL" in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config.json"))):
#     logging.log(logging.INFO, "[CHAT] Using OpenAI API")
import os
import openai
openai.api_base = "http://localhost:8080/v1"
openai.api_key = "sx-xxx"
OPENAI_API_KEY = "sx-xxx"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
logging.log(logging.INFO, "[CHAT] Using local OpenAI API")

# Don't ask me why I wrote these prompts, I don't know either. I just know that they work.
# Maybe they're just there to make the chatbot seem more human, and respond better to questions.
messages = [ 
    {"role": "system", "content": "You are a intelligent assistant. Speak in English only. Give short responses, enough to be spoken in 5 to 8 senconds. you are a chatbot at a science fair. When someone asks you where a person is, respond according to the data given below in json (just tell the place their project is at)."},
    {"role": "system", "content": "The speech recognition systems used in this project are not perfect. Allow for some errors in the speech recognition. For example, Vrunda may be recognized as Brenda or Vrinda. You can use the json data given below to figure it out. If you are not sure, just say you don't know."},
    {"role": "system", "content": "If there are multiple projects with the same name, just say a random one, then tell them that there are multiple projects with the same name, and ask them to be more specific, then tell them where the project theyre talking about is located."},
    # {"role": "system", "content": "You are a chatbot at a science fair. When someone asks you where a person is, respond according to the data given below in json."},
    # {"role": "system", "content": "You must be able to answer questions about the science fair from the json data given below:"},
]
# with open("Chatbot/expo_data.json", "r") as f:
#     expo_data = json.load(f)
# # Append all of them one by one
# for floor in expo_data["projects"]:
#     messages.append({"role": "system", "content": "The project {} is located on the {} floor, made by {}, of class {}.".format(
#         floor["title"],
#         floor["floor"],
#         floor["roomNumber"],
#         str(floor["members"]).replace("'", "").replace("[", "").replace("]", ""),
#         floor["class"]
#     )})
def get_response(message):
    global messages
    messages.append(
        {"role": "user", "content": message},
    )
    completion = openai.ChatCompletion.create(
        model="koala-7B.ggmlv3.q2_K.bin",
        messages=messages,
        max_tokens=16,
        temperature=0.7,
    )
    ret = str(completion.choices[0].message.content)
    if ":" in ret:
        ret = ret.split(":")[1]
    return ret
try:
    get_response("What is AI?")
    logging.log(logging.INFO, "[CHAT] OpenAI API is working!")
except Exception as e:
    logging.log(logging.INFO, "[CHAT] OpenAI API is NOT working! {}".format(e))

class ChatBot:
    def __init__(self, speaker=None):
        logging.log(logging.INFO, "[CHAT] ChatBot __init__")
        self.conversation_data = []
        self.fallbacks = []
        # self.cache = {}
        # self.nlp_cache = {}
        # if not os.path.exists("./cache.json"):
        #     with open("./cache.json", "w") as f:
        #         f.write("{}")
        self.loader = yaml.SafeLoader
        self.speaker = speaker
        self.nlp = spacy.load("en_core_web_lg")
        
    def train(self, conversation_data):
        logging.log(logging.INFO, f"[CHAT] Training chatbot on {len(conversation_data)} conversation data points...")
        self.conversation_data += conversation_data
        self.save_hash = sha256.sha256(str(conversation_data).encode()).hexdigest()
        
    def create_offline_cache(self):
        logging.log(logging.INFO, "[CHAT] Creating text to speech cache...")
        for data in tqdm.tqdm(self.conversation_data, desc="Creating tts cache for train"):
            for utterance in data:
                if self.speaker and not self.is_question(utterance):
                    self.speaker.create_speech_cache(utterance)
        for utterance in tqdm.tqdm(self.fallbacks, desc="Creating tts cache for fallbacks"):
            if self.speaker and not self.is_question(utterance):
                self.speaker.create_speech_cache(utterance)
    
    def is_question(self, utterance):
        return "?" in utterance
    
    def train_fallbacks(self, fallbacks):
        logging.log(logging.INFO, f"[CHAT] Training chatbot on {len(fallbacks)} fallback data points...")
        self.fallbacks += fallbacks
    
    def calculate_similarity_dirty(self, a, b):
        val = self.fuzz_ratio(a.lower(), b.lower())
        if val > THRESHOLD:
            return 0
        else:
            return val
    
    def calculate_similarity_better(self, a, b):
        if not a in self.nlp_cache:
            self.nlp_cache[a] = self.nlp(' '.join([str(token) for token in self.nlp(a.lower()) if not token.is_stop]))
        if not b in self.nlp_cache:
            self.nlp_cache[b] = self.nlp(' '.join([str(token) for token in self.nlp(b.lower()) if not token.is_stop]))
        return self.nlp_cache[a].similarity(self.nlp_cache[b])
    
    def calculate_similarity(self, query, conversation_entry):
        similarity_scores = []
        for utterance in conversation_entry:
            similarity_score = self.calculate_similarity_dirty(query, utterance) #+ self.calculate_similarity_better(query, utterance)
            # TODO: Make nlp similarity better
            similarity_scores.append(similarity_score)
        return similarity_scores
    
    def answer(self, query):
        if query == "":
            return ""
        logging.log(logging.INFO, f"[CHAT] Answering query: {query}")
        logging.log(logging.INFO, "[CHAT] Calculating similarities...")
        similarities = []
        for conversation_entry in tqdm.tqdm(self.conversation_data, desc="Calculating similarities"):
            similarities.append(self.calculate_similarity(query, conversation_entry))
        logging.log(logging.INFO, "[CHAT] Similarities calculated. Linearizing...")
        
        linear_similarities = []
        for i, similarity_scores in enumerate(similarities):
            for j, score in enumerate(similarity_scores):
                if score > THRESHOLD:
                    linear_similarities.append((score, (i, j)))
        logging.log(logging.INFO, "[CHAT] Linearized. Sorting...")
        self.cache[query] = linear_similarities

        self.save_cache()

        try:
            logging.log(logging.INFO, "[CHAT] Sorted matches. Finding max...")
            max_similarity = max(i[0] for i in linear_similarities)
            max_similarity_index = next(i[1] for i in linear_similarities if i[0] == max_similarity)
            logging.log(logging.INFO, f"[CHAT] Max found to be {max_similarity} at index {max_similarity_index}")
            global messages
            messages.append({"role": "user", "content": query})
            messages.append({"role": "system", "content": self.conversation_data[max_similarity_index[0]][max_similarity_index[1]]})
            return self.conversation_data[max_similarity_index[0]][max_similarity_index[1] + 1]
        except:
            try:
                logging.log(logging.INFO, "[CHAT] No matches found. Trying ChatGPT...")
                return get_response(query)
            except Exception as e:
                logging.log(logging.INFO, f"[CHAT] ChatGPT failed with {e}. Using random fallback...")
                return self.random_fallback()
        
    def random_fallback(self):
        logging.log(logging.INFO, "[CHAT] Random fallback!")
        return random.choice(self.fallbacks)
    
    def train_expo_data(self, expo_data):
        with open(expo_data, "r") as f:
            expo_data = json.load(f)
        data = []
        
        logging.log(logging.INFO, "[CHAT] Training chatbot on categories...")
        
        # When there are more that 2 projects with the same name
        found = {}
        for project in expo_data["projects"]:
            try:
                found[project["title"]] += 1
            except KeyError:
                found[project["title"]] = 1
        found_exceptions = [i for i in found if found[i] > 1]
        for exception in found_exceptions:
            print(exception)
            
        # Questions about projects
        for project in expo_data["projects"]:
            # Where is project X?
            whereis_questions = [
                "Where is project {}?".format(project["title"]),
                "Where is {}?".format(project["title"]),
                "Where is {} located?".format(project["title"]),
                "Where is {} located at?".format(project["title"]),
                "Where is {} located in the expo?".format(project["title"]),
                "Where can I find {}?".format(project["title"]),
                "Where can I find {} in the expo?".format(project["title"]),
                "Where can I find {} located?".format(project["title"]),
                "Where can I find {} located at?".format(project["title"]),
                "Where is the project {}?".format(project["title"]),
                "Where is the {}?".format(project["title"]),
                "Where is the {} located?".format(project["title"]),
                "Where is the {} located at?".format(project["title"]),
            ]
            for question in whereis_questions:
                data.append([
                    question,
                    "The project {} is located on the {} floor, room {}.".format(
                        project["title"],
                        self.numerify(project["floor"]),
                        self.number_to_speech(project["roomNumber"])
                    )
                ])
            
            # Who made project X?
            whois_questions = [
                "Who made project {}?",
                "Who made {}?",
                "Who made {} project?",
                "Who made the project {}?",
                "Who made the {}?",
                "Who made the {} project?",
            ]
            for question in whois_questions:
                data.append([
                    question,
                    "The project {} was made by {}.".format(
                        project["title"],
                        self.mems2str(project["members"])
                    )
                ])
                
            # Student X made what project?
            for member in project["members"]:
                studentmade_questions = [
                    "{} made what project?",
                    "{} made what?",
                    "{} made what project?",
                    "{} made what project?",
                    "What project did {} make?",
                    "What did {} make?",
                ]
                for question in studentmade_questions:
                    data.append([
                        question.format(member),
                        "{} made the project {}.".format(
                            member,
                            project["title"]
                        )
                    ])
                
            # Where is student X?
            for member in project["members"]:
                whereis_questions = [
                    "Where is {}?",
                    "Where can I find {}?",
                    "I want to meet {}",
                ]
                for question in whereis_questions:
                    data.append([
                        question.format(member),
                        "The student {} is located on the {} floor, room {}.".format(
                            member,
                            self.numerify(project["floor"]),
                            self.number_to_speech(project["roomNumber"])
                        )
                    ])
                    
            # Where is project X by student Y?
            for member in project["members"]:
                whereis_questions = [
                    "Where is {}'s project?",
                    "Where can I find {}'s project?",
                    "{}'s project"
                ]
                for question in whereis_questions:
                    data.append([
                        question.format(member),
                        "The project {} is located on the {} floor, room {}.".format(
                            project["title"],
                            self.numerify(project["floor"]),
                            self.number_to_speech(project["roomNumber"])
                        )
                    ])

            # Which class made project X?
            if len(project["class"]) > 1:
                whichclass_questions = [
                    "Which class made project {}?",
                    "Which class made {}?",
                    "Which class made {} project?",
                    "Which class made the project {}?",
                    "Which class made the {}?",
                    "Which class made the {} project?",
                ]
                for question in whichclass_questions:
                    try:
                        data.append([
                            question,
                            "The project {} was made by {}.".format(
                                project["title"],
                                self.number_to_speech(project["class"].split(" ")[0]) + " " + project["class"].split(" ")[1]
                            )
                        ])
                    except IndexError:
                        pass
                    
            # Who is X?
            for member in project["members"]:
                whois_questions = [
                    "Who is {}?",
                    "Who is {}?",
                    "Who is {}?",
                    "Who is {}?",
                ]
                for question in whois_questions:
                    data.append([
                        question.format(member),
                        "{} is a member of the project {}.".format(
                            member,
                            project["title"]
                        )
                    ])
            
        # What are the projects in room X?
        for room in expo_data["rooms"]:
            whatare_questions = [
                "What are the projects in room {}?",
                "What are the other projects in room {}?",
                "What can I find in room {}?",
                "What can I see in room {}?",
                "What else is in room {}?",
                "What else can I see in room {}?",
                "What else can I find in room {}?",
            ]
            found = []
            for project in expo_data["projects"]:
                if project["roomNumber"] == room:
                    found.append(project["title"])
            for question in whatare_questions:
                data.append([
                    question.format(self.number_to_speech(room)),
                    "The projects in room {} are: {}".format(
                        self.number_to_speech(room),
                        self.mems2str(found)
                    )
                ])
            
        self.train(data)
        
    def mems2str(self, members):
        if len(members) == 1:
            return members[0]
        elif len(members) == 2:
            return members[0] + " and " + members[1]
        else:
            return ", ".join(members[:-1]) + ", and " + members[-1]
        
    def numerify(self, number):
        if number == str(1):
            return "ground"
        elif number == str(2):
            return "first"
        elif number == str(3):
            return "second"
        elif number == str(4):
            return "third"
        
    def number_to_speech(self, number):
        """
        Convert 16 into sixteen, etc.
        """
        number = str(number)
        number = list(number)
        for i in range(len(number)):
            number[i] = p.number_to_words(number[i])
        return " ".join(number)

    def get_category_index(self, expo_data, category):
        for i in range(len(expo_data["categories"])):
            if expo_data["categories"][i]["title"] == category:
                return i
        return None
    
    def fuzz_ratio(self, a, b):
        return fuzz.ratio(a, b)
    
    def load_cache(self):
        logging.log(logging.INFO, "[CHAT] Loading cache...")
        try:
            with open("cache.json", "r") as f:
                self.cache = json.load(f)
                try:
                    if self.cache["train_data_hash"] != self.save_hash:
                        self.cache = {
                            "train_data_hash": self.save_hash,
                        }
                        self.save_cache()
                except KeyError:
                    self.cache = {
                        "train_data_hash": self.save_hash,
                    }
                    self.save_cache()
        except FileNotFoundError:
            self.cache["train_data_hash"] = ""
            self.save_cache()
    
    def save_cache(self):
        logging.log(logging.INFO, "[CHAT] Saving cache...")
        try:
            with open("cache.json", "w") as f:
                if not "train_data_hash" in self.cache:
                    self.cache["train_data_hash"] = self.save_hash
                json.dump(self.cache, f)
        except FileNotFoundError:
            self.cache = {}
            
    def train_from_yaml(self, yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.load(f, Loader=self.loader)["conversations"]
        logging.log(logging.INFO, "Training chatbot on yaml file: {}".format(yaml_file))
        self.train(data)
    
    def train_from_corpus(self, corpus_dir, include=["*"]):
        logging.log(logging.INFO, "[CHAT] Training chatbot on corpus directory: {}".format(corpus_dir))
        for filename in os.listdir(corpus_dir):
            if include[0] == "*":
                if filename.endswith(".yml"):
                    self.train_from_yaml(os.path.join(corpus_dir, filename))
            else:
                if filename.split(".")[0] in include:
                    self.train_from_yaml(os.path.join(corpus_dir, filename))