import re
import os
import time
import json
import torch
import random
import openai
import shutil
import pickle
from ast import literal_eval
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import functools
import logging
import inspect
from dotenv import load_dotenv
from transformers import pipeline

from src.processing.neo4j_operations import DialogueGraphPersister
from src.processing.neo4j_operations import Neo4jGraph
from src.paths import LOCAL_RAW_DATA_PATH
from src.config import device
from src.utils import extract_bot_reply
from src.processing.relation_extraction_evaluator import parse_json_objects

load_dotenv()  # take environment variables from .env.

USER_NAME = 'Hilde'
BOT_NAME = 'Adele'

# LLM_NAME = 'Open-Orca/Mistral-7B-OpenOrca'
LLM_NAME = 'HuggingFaceH4/zephyr-7b-beta'
# LLM_NAME = 'gpt-3.5-turbo' 
LANG = 'de' # 'en'

DATASET_NAME = 'chatgpt_pipeline'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NEO4J_URI=os.environ.get('NEO4J_URI')
NEO4J_USERNAME=os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD=os.environ.get('NEO4J_PASSWORD')

# Create a separate logger
logger = logging.getLogger('chatbot_interactions')
handler = logging.FileHandler(LOCAL_RAW_DATA_PATH / 'chatbot_interactions.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


if 'gpt' not in LLM_NAME:
    pipe = pipeline("text-generation", model=LLM_NAME, torch_dtype=torch.bfloat16, device_map=device)



def make_variables_colorful_html(txt):
    tmp_txt = txt.replace('{{', '__1').replace('}}', '__2')
    tmp_txt = re.sub(r'\{(.*?)\}', r'<span class="text-primary">{\1}</span>', tmp_txt )
    tmp_txt = tmp_txt.replace('__1', '{{').replace('__2', '}}')
    return tmp_txt

def log_interaction(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the calling class and method
        stack = inspect.stack()
        calling_class = stack[1][0].f_locals["self"].__class__.__name__
        calling_method = stack[1][0].f_code.co_name

        log_entry = {
            'calling_class_method': f"{calling_class}.{calling_method}",
            'inputs': kwargs if kwargs else args[1:],  # Skip 'self' in args
        }
        result = func(*args, **kwargs)
        log_entry['output'] = result
        logger.info(json.dumps(log_entry, indent=4, sort_keys=True, ensure_ascii=False))
        return result
    return wrapper

class Message:
    def __init__(self, role, content, timestamp=None):
        self.role = role
        self.content = content
        if not timestamp:
            timestamp = time.time()
        self.timestamp = timestamp

    def to_dict(self, idx=None, drop_timestamp=False):

        # Determine if names should be prepended.
        prepend_names = 'gpt' not in LLM_NAME and idx not in (0, None)
        
        # Prepend the appropriate name based on the role and prepend_names flag.
        prepend = f"{USER_NAME}: " if self.role == 'user' and prepend_names else ''
        prepend = f"{BOT_NAME}: " if self.role != 'user' and prepend_names else prepend

        # Set the role to 'assistant' if the role is not 'user' and idx is positive.
        role = 'assistant' if self.role != 'user' and idx and idx > 0 else self.role

        # Create the base dictionary.
        base_dict = {"role": role, "content": f"{prepend}{self.content}"}

        # Add timestamp if not dropped.
        if not drop_timestamp:
            base_dict["timestamp"] = self.timestamp

        return base_dict
        
class DialogueLogger:
    def __init__(self, output_dir, dataset_name=DATASET_NAME):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.persister = DialogueGraphPersister("my_dataset")

    def save_message(self, message: Message, turn):
        message_data = {
            "turn": turn,
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,  # Timestamp in seconds since the Epoch
        }
        file_name = f"{str(message_data['turn']).zfill(3)}_{message.role}_{int(message_data['timestamp'])}.json"
        file_path = os.path.join(self.output_dir, file_name)
        with open(file_path, 'w') as file:
            json.dump(message_data, file, indent=2)

    def load_chat_history(self):
        files = sorted(os.listdir(self.output_dir)) 
        history = []
        for file in files:
            if file.endswith('.json'):  # To ensure we only process JSON files
                file_path = os.path.join(self.output_dir, file)
                with open(file_path, 'r') as f:
                    message_data = json.load(f)
                    history.append(Message(message_data['role'], message_data['content'], message_data['timestamp']))
        return history

    def archive_dialogue_logs(self):
        # Create a folder to store archived logs
        archive_folder = os.path.join(self.output_dir, "archive")
        os.makedirs(archive_folder, exist_ok=True)

        # Define the name of the archive folder for this specific dialogue
        timestamp = int(time.time())  # Timestamp in seconds since the Epoch
        human_readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
        archive_subfolder = os.path.join(archive_folder, f"{USER_NAME}_{BOT_NAME}_{human_readable_time}")

        # Create a folder for the archive of this dialogue
        os.makedirs(archive_subfolder, exist_ok=True)

        # Move all dialogue log files to the archive folder
        files = os.listdir(self.output_dir)
        for file in files:
            if file.endswith('.json'):  # To ensure we only process JSON files
                file_path = os.path.join(self.output_dir, file)
                new_file_path = os.path.join(archive_subfolder, file)
                os.rename(file_path, new_file_path)
                
        self.persister.check_and_archive(os.path.join(archive_subfolder, "neo4j_dump"))
        self.persister.close_connection()
    
    def get_archive_folders(self):
        archive_folder = os.path.join(self.output_dir, "archive")

        try:
            return os.listdir(archive_folder)
        except FileNotFoundError:
            return []    

class TemplateBasedGPT:
    def __init__(self, template_path=None, model=LLM_NAME, api_key=OPENAI_API_KEY, debug=False):
        self.model = model
        self.debug = debug
        openai.api_key = api_key

        if template_path:
            with open(file=template_path, encoding='utf8') as fp:
                self.template = fp.read()
            
        if 'gpt' not in model:
            self.pipe = pipe

    @log_interaction
    def generate_chatbot_response(self, model, messages, max_tokens=256, bot_name=None):
        if 'gpt' in model:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            relationships = response['choices'][0]['message']['content']
            return relationships
        else:
            prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = self.pipe(prompt, max_new_tokens=256,
                               do_sample=True, temperature=0.01, top_k=100, top_p=0.99)
            raw_response = extract_bot_reply(outputs, bot_name=bot_name)
            return raw_response

    def extract(self, input_text, max_token):
        # This function is meant to be overwritten in child classes
        pass

class GPTTripletExtractor(TemplateBasedGPT):
    def __init__(self, template_path=LOCAL_RAW_DATA_PATH / f'prompt-templates/{LANG}/triplet-extraction.txt', model=LLM_NAME, api_key=OPENAI_API_KEY, debug=False):
        super().__init__(template_path, model, api_key, debug)

    def extract(self, input_text, max_token=256):
        input_dialogue_str = json.dumps(input_text, indent=2)
        messages = [{"role": "system", "content": self.template.format(input_dialogue=input_dialogue_str)}]

        if self.debug:
            return [{'x': 'x_DEBUG', 'x_type': 'type_DEBUG', 'y': 'y_DEBUG', 'y_type': 'type_DEBUG', 'r': 'r_DEBUG'}]
        
        raw_inference = self.generate_chatbot_response(self.model, messages, max_token)
        
        try:
            relationships = parse_json_objects(raw_inference.split('Input:')[0])
        except Exception as e:
            print(f'Exception thrown: {e}')
            relationships = []
            
        return relationships

class OpenerGenerator:
    def __init__(self, user_name, bot_name, state_path_dir, anamnese_mode=False, lang=LANG):
        self.user_name = user_name
        self.bot_name = bot_name
        self.anamnese_mode = anamnese_mode
        if self.anamnese_mode:
            self.state_path = state_path_dir / f"opener_state_anamnese_{lang}.pkl"
        else:
            self.state_path = state_path_dir / f"opener_state_{lang}.pkl"
        
        if lang == 'en':
            self.all_items = {
                "greetings": [
                    f"Hello, {self.user_name}, it's {self.bot_name} here!",
                    f"Hi, {self.user_name}, this is {self.bot_name}!",
                    f"Good day, {self.user_name}! It's {self.bot_name} here.",
                    f"{self.bot_name} here, hi {self.user_name}!"
                ],
                "availability_requests": [
                    "Can we talk now?",
                    "Do you want a quick chat?",
                    "Are you free to talk now?",
                ],
                "topic_introductions": [
                    # "I was thinking, what kinds of songs do you like?", # out of data schema
                    # "I wanted to know, do you enjoy reading?", # out of data schema
                    # "I was wondering, what's the last movie you loved?", # out of data schema
                    "I'm interested in knowing how you're feeling about your medications.",    
                    "Tell me about someone dear to you. I'd love to get to know them!",  
                    "Tell me about your last trip. I'd love to hear it!",  
                    "Tell me about a cherished memory of yours. I'd love to hear it!",    
                    "I just wanted to hear from you!",
                    "I was curious, do you have any pets?",
                    "I wanted to know, how's your back doing?",
                    "I was wondering, what was the place you last visited?",
                ]
            }
            
            if self.anamnese_mode:
                self.all_items['topic_introductions'] = [
                    "I want to know about your medical history", # out of data schema
                ]
        elif lang == 'de':
            self.all_items = {
                "greetings": [
                    f"Hallo, {self.user_name}, hier ist {self.bot_name}!",
                    f"Hi, {self.user_name}, hier spricht {self.bot_name}!",
                    f"Guten Tag, {self.user_name}! Hier ist {self.bot_name}.",
                    f"{self.bot_name} hier, hallo {self.user_name}!"
                ],
                "availability_requests": [
                    "Können wir jetzt sprechen?",
                    "Möchtest du kurz plaudern?",
                    "Hast du jetzt Zeit für ein Gespräch?",
                ],
                "topic_introductions": [
                    "Ich möchte gerne wissen, wie es dir mit deinen Medikamenten geht.",
                    "Erzähl mir von jemandem, der dir lieb ist. Ich würde gerne mehr darüber erfahren!",
                    "Erzähl mir von deiner letzten Reise. Ich würde sie gerne hören!",
                    "Erzähl mir von einer geschätzten Erinnerung von dir. Ich würde sie gerne hören!",
                    "Ich wollte einfach mal von dir hören!",
                    "Ich war neugierig, hast du Haustiere?",
                    "Ich wollte wissen, wie geht es deinem Rücken?",
                    "Ich habe mich gefragt, welches der letzte Ort war, den du besucht hast?",
                ]
            }
            
            if self.anamnese_mode:
                self.all_items['topic_introductions'] = [
                    "Ich möchte deine Krankengeschichte wissen",
            ]
        else:
            raise NotImplementedError

        try:
            with open(self.state_path, 'rb') as f:
                self.available_items = pickle.load(f)
        except (FileNotFoundError, EOFError):
            self.available_items = self.all_items.copy()

    def save_state(self):
        with open(self.state_path, 'wb') as f:
            pickle.dump(self.available_items, f)

    def get_item(self, category):
        if not self.available_items[category]:
            # all items have been used, so repopulate the list
            self.available_items[category] = self.all_items[category].copy()

        item = random.choice(self.available_items[category])
        self.available_items[category].remove(item)

        return item

    def generate_opener(self):
        greeting = self.get_item("greetings")
        availability_request = self.get_item("availability_requests")
        topic_introduction = self.get_item("topic_introductions")

        # Save the current state
        self.save_state()

        return f"{greeting} {availability_request} {topic_introduction}"

class MemoryOpenerGenerator(TemplateBasedGPT):
    def __init__(self, user_name, bot_name, template_path=LOCAL_RAW_DATA_PATH / f'prompt-templates/{LANG}/follow-up-message.txt', model=LLM_NAME, api_key=OPENAI_API_KEY, debug=False):
        super().__init__(template_path, model, api_key, debug)
        self.user_name = user_name
        self.bot_name = bot_name
        self.memory_puller = Neo4jMemoryPuller(user_name=self.user_name)
    
    def generate_opener(self, max_turn_history=15, use_sliding_window=True, max_token=60):
        topic, strategy, relations, dialogue = self.memory_puller.pull_memory()

        if use_sliding_window:
            selected_dialogues = self._select_sliding_window_dialogues(dialogue, max_turn_history)
        else:
            selected_dialogues = self._select_recent_past_dialogues(dialogue, max_turn_history)

        dialogue_str = json.dumps(selected_dialogues, indent=2)
        relations_str = json.dumps(relations, indent=1)

        template = self.template.format(
            bot_name=self.bot_name,
            user_name=self.user_name,
            topic=topic,
            relation_list=relations_str,
            chat_history=dialogue_str)

        messages = [{"role": "system", "content": template}]

        if self.debug:
            return "Debugging message", topic, strategy, relations_str, dialogue_str

        opener = self.generate_chatbot_response(self.model, messages, max_token)
        
        return opener, topic, strategy, relations_str, dialogue_str

    def _select_sliding_window_dialogues(self, dialogues, max_turn_history):
        # Randomly select a starting index for the section
        max_start_index = max(len(dialogues) - max_turn_history, 0)
        start_index = random.randint(0, max_start_index)

        # Extract the section of dialogues
        selected_dialogues = dialogues[start_index:start_index + max_turn_history]
        return selected_dialogues

    def _select_recent_past_dialogues(self, dialogues, max_turn_history):
        # Grab the most recent past (last n turns)
        selected_dialogues = dialogues[-max_turn_history:]
        return selected_dialogues

        
class Neo4jMemoryPuller(Neo4jGraph):
    def __init__(self, user_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_name = user_name
        self.topics = ['people', 'places', 'pet']
        self.relations = {
            "people": ["acquaintance", "children", "other_family", "parents", "siblings", "spouse"],
            "places": ["place_of_residence", "visited_place", "residents_of_place", "visitors_of_place"],
            "pet": ["pet"]
        }
        self.node_types = {
            "people": ["PERSON"],
            "places": ["ORG", "GPE"],
            "pet": ["ANIMAL"]
        }
        self._update_relations_and_types()
    
    @staticmethod
    def _flatten_unique(dialogue_dict):
        seen_items = set()
        sorted_dialogue_list = [d['text'] for d in sorted(dialogue_dict, key=lambda x: x['dialogue_id'])]
        flattened_list = [item for sublist in sorted_dialogue_list for item in sublist if not (item in seen_items or seen_items.add(item))]
        return flattened_list

    def _update_relations_and_types(self):
        # Fetch all present relationship names and entity labels from the graph
        with self.driver.session() as session:
            result_rel = session.run("MATCH ()-[r]-() where type(r) = 'RELATION' RETURN DISTINCT r.type;").values()
            rel_types = [item for sublist in result_rel for item in sublist]

            result_label = session.run("MATCH (n:Entity) RETURN DISTINCT n.type;").values()
            node_labels = [item for sublist in result_label for item in sublist]

        # Update self.relations and self.node_types to only contain existing labels/types
        for topic in self.topics:
            self.relations[topic] = [r for r in self.relations[topic] if r in rel_types]
            self.node_types[topic] = [t for t in self.node_types[topic] if t in node_labels]
            
    def pull_memory(self, topic=None, strategy=None):
        if not topic:
            topic = random.choice(self.topics)
        if not strategy:
            strategies = ['relation', 'type']
            strategy = random.choice(strategies)
        
        # Ensure the selected topic has available data
        while not self.relations[topic] and not self.node_types[topic]:
            self.topics.remove(topic)
            if not self.topics:
                print("No available topics to pull data from.")
                return topic, strategy, None, None
            topic = random.choice(self.topics)

        if strategy == 'relation' and self.relations[topic]:
            relation = random.choice(self.relations[topic])
            path, dialogue = self._pull_data_for_relation(relation)
        elif strategy == 'type' and self.node_types[topic]:
            node_type = random.choice(self.node_types[topic])
            path, dialogue = self._pull_data_for_node_type(node_type)
        else:
            print(f"No {strategy} found for topic '{topic}'")
            path, dialogue = None, None
        
        return topic, strategy, path, dialogue

    def _pull_data_for_relation(self, relation_type):
        query = (
            f"MATCH p=(:Entity {{name:'{self.user_name}'}})-[*..3]->(e:Entity) "
            f"WHERE ANY (rel IN relationships(p) WHERE rel.type = '{relation_type}') "
            f"WITH p AS path "
            f"ORDER BY length(path) ASC "
            f"WITH collect(path) AS paths, length(collect(path)[0]) as minLength "
            f"UNWIND paths AS minimalPaths "
            f"WITH minimalPaths "
            f"WHERE length(minimalPaths) = minLength "
            f"MATCH (d:Dialogue)-[r]-(m) WHERE m IN nodes(minimalPaths) "
            f"RETURN minimalPaths as path, collect(distinct {{text:d.text, dialogue_id: d.id}}) AS dialogue LIMIT 3"
        )

        with self.driver.session() as session:
            results = list(session.run(query))
            result = random.choice(results)
            path, dialogue = result['path'], result['dialogue']
            return self.convert_path(path), self._flatten_unique(dialogue)
        
    def _pull_data_for_node_type(self, node_type):
        query = (
            f"MATCH p=(:Entity {{name: '{self.user_name}'}})-[*..3]->(e:Entity {{type: '{node_type}'}}) "
            f"WITH p AS path "
            f"ORDER BY length(path) ASC "
            f"WITH collect(path) AS paths, length(collect(path)[0]) as minLength "
            f"UNWIND paths AS minimalPaths "
            f"WITH minimalPaths "
            f"WHERE length(minimalPaths) = minLength "
            f"MATCH (d:Dialogue)-[r]-(m) WHERE m IN nodes(minimalPaths) "
            f"RETURN minimalPaths as path, collect(distinct {{text:d.text, dialogue_id: d.id}}) AS dialogue LIMIT 3"
        )

        with self.driver.session() as session:
            results = list(session.run(query))
            result = random.choice(results)
            path, dialogue = result['path'], result['dialogue']
            return self.convert_path(path), self._flatten_unique(dialogue)


class ChatGPT(TemplateBasedGPT):
    def __init__(self,
                 bot_name=BOT_NAME,
                 user_name=USER_NAME,
                 model=LLM_NAME,
                 api_key=OPENAI_API_KEY,
                 debug=False,
                 output_dir=LOCAL_RAW_DATA_PATH / 'dialogue_logs',
                 dataset_name=DATASET_NAME,
                 anamnese_mode=False):

        super().__init__(None, model, api_key, debug)
        self.model = model
        self.debug = debug
        openai.api_key = api_key
        self.bot_name = bot_name
        self.user_name = user_name
        self.anamnese_mode = anamnese_mode
        if anamnese_mode:
            preprompt_template_path=LOCAL_RAW_DATA_PATH / f"prompt-templates/{LANG}/AnamneseBot/chat-pre-prompt.txt"
            triplet_preprompt_path=LOCAL_RAW_DATA_PATH / f'prompt-templates/{LANG}/AnamneseBot/triplet-extraction.txt'
        else:
            preprompt_template_path=LOCAL_RAW_DATA_PATH / f"prompt-templates/{LANG}/chat-pre-prompt-002.txt"
            triplet_preprompt_path=LOCAL_RAW_DATA_PATH / f'prompt-templates/{LANG}/triplet-extraction.txt'

        self.relationship_extractor = GPTTripletExtractor(template_path=triplet_preprompt_path,api_key=api_key, model=model, debug=debug)
        self.graph_persister = DialogueGraphPersister(dataset_name)  
        self.dialogue_logger = DialogueLogger(output_dir)  # Add this line to instantiate the DialogueLogger
        self.preprompt_path = preprompt_template_path
        with open(preprompt_template_path, encoding='utf8') as fp:
            self.preprompt_template = fp.read()
                
        self.load_chat_history()

        # Initialize the OpenerGenerator
        self.opener_generator = OpenerGenerator(self.user_name, self.bot_name, output_dir, self.anamnese_mode)
        self.memory_opener = MemoryOpenerGenerator(self.user_name, self.bot_name, debug=debug)
        agenda_topics = ['Any exams showing bad kidneys? If yes, ask when', 'Exams indicate high blood pressure? If yes, ask when', 'Ever had blood in urine? If yes, ask when']
        if not self.history:
            preprompt = self.preprompt_template.format(bot_name=self.bot_name, user_name=self.user_name, agenda_topics=str(agenda_topics))
            self.add_and_log_message('system', preprompt)

    def load_chat_history(self):
        self.history = self.dialogue_logger.load_chat_history()

    def add_and_log_message(self, role, content):
        message = Message(role, content)
        self.history.append(message)
        self.dialogue_logger.save_message(message, len(self.history))
        return message
    
    def trim_history(self, num_last_msgs=15):
        # If there are more than (num_last_msgs + 1) messages in the history
        if len(self.history) > num_last_msgs + 1:
            # Keep only the first message and the last num_last_msgs
            history = [self.history[0]] + self.history[-num_last_msgs:]
        else:
            # Otherwise, use the entire history
            history = self.history
        return history

    def generate_and_add_response(self, max_token=60):
        # If debug mode is enabled, don't call the API
        if self.debug:
            return self.add_and_log_message("system", "Debugging message")

        num_last_msgs=15 if 'gpt' in self.model else 5
            
        # Trim the history
        history = self.trim_history(num_last_msgs)
        messages = [msg.to_dict(drop_timestamp=True, idx=idx) for idx, msg in enumerate(history)]

        # Extract the chatbot's message from the response
        chatbot_message = self.generate_chatbot_response(self.model, messages, max_token, self.bot_name)

        return self.add_and_log_message("system", chatbot_message)

    def format_role(self, role):
        if role == 'system':
            return self.bot_name
        else:
            return self.user_name

    @staticmethod
    def preprocess_relations(relations):
        for relation in relations:
            for key, value in relation.items():
                if "_type" in key:
                    relation[key] = value.upper()
                elif key == "r":
                    if isinstance(value, str):
                        relation[key] = value.lower()
                    elif isinstance(value, list):
                        relation[key] = [item.lower() for item in value]
        return relations

    def dump_to_neo4j(self, dialogue, predicted_relations):
        preprocessed_relations = self.preprocess_relations(predicted_relations)
        self.graph_persister.process_dialogue(dialogue, preprocessed_relations)
        self.graph_persister.close_connection()

    def extract_triplets(self, n_last_turns=5, dump_graph=True):
        # Prepare the last n turns as a list
        last_n_turns = [f"{self.format_role(msg.role)}: {msg.content}" for msg in self.history[1:][-n_last_turns:]]

        # Extract relationships
        relationships = self.relationship_extractor.extract(last_n_turns)
        
        if dump_graph and relationships:
            self.dump_to_neo4j(last_n_turns, relationships)
        return relationships
    
    def start_conversation(self):
        print("DEBUG=",self.debug)
        initial_prompt = self.opener_generator.generate_opener()
        self.add_and_log_message("system", initial_prompt)

        print(f"{self.bot_name}: {initial_prompt}")

        # Conversation loop
        while True:
            user_input = input(f"{self.user_name}: ")

            # Exit conversation if user types 'quit'
            if user_input.lower() == 'quit':
                break

            self.add_and_log_message("user", user_input)
            
            # Analyze the conversation history after each user input
            relationships = self.extract_triplets()
            # print(f"# EXTRACTED TRIPLETS: {relationships}")
            
            response = self.generate_and_add_response()
            print(f"{self.bot_name}: {response.content}")

    def reload_from_archive(self, archive_subfolder):
        archive_path = os.path.join(self.dialogue_logger.output_dir, "archive", archive_subfolder)
        output_dir = self.dialogue_logger.output_dir

        # Step 1: Load the chat history
        self.history = []
        for file in os.listdir(archive_path):
            if file.endswith('.json'):
                file_path = os.path.join(archive_path, file)
                with open(file_path, 'r') as f:
                    message_data = json.load(f)
                    self.history.append(Message(message_data['role'], message_data['content'], message_data['timestamp']))

        # Step 2: Delete all JSON files in the output directory
        for file in os.listdir(output_dir):
            if file.endswith('.json'):
                file_path = os.path.join(output_dir, file)
                os.remove(file_path)

        # Step 3: Copy all files from the archive folder to the output directory
        for file in os.listdir(archive_path):
            if file.endswith('.json'):
                src_path = os.path.join(archive_path, file)
                dst_path = os.path.join(output_dir, file)
                shutil.copy(src_path, dst_path)

        # Step 4: Load the graph data
        self.graph_persister.load_archived_data(archive_path)


def load_chat_history(output_dir, max_files=50):
    # @TODO: check out why old sorting was nenecessary
    # files = sorted(os.listdir(output_dir), key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))  #
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.json')])
    history = []
    for file in files[-max_files:]:  # Only process the last max_files files
        file_path = os.path.join(output_dir, file)
        with open(file_path, 'r') as f:
            message_data = json.load(f)
            if message_data.get('turn') == 1:
                continue # Skip preprompt message
            history.append(Message(message_data['role'], message_data['content'], message_data['timestamp']))
    return history


app = Flask(__name__)
@app.route('/')
def home():
    output_dir = LOCAL_RAW_DATA_PATH / 'dialogue_logs'
    history = load_chat_history(output_dir, max_files=50)
    # Format history so that it can be easily processed in the template
    formatted_history = [msg.to_dict(idx=None) for idx, msg in enumerate(history)]
    
    debug_mode = request.args.get('debug') == 'true'  # Get debug mode from the URL parameters
    anamnese_mode = request.args.get('anamnese') == 'true'  # Get debug mode from the URL parameters
    chat_gpt = ChatGPT(model=LLM_NAME, debug=debug_mode, anamnese_mode=anamnese_mode)
    
    prompt_templates = {
        'chat_instructions': make_variables_colorful_html(chat_gpt.preprompt_template),
        'triplet_extraction': make_variables_colorful_html(chat_gpt.relationship_extractor.template),
        'memory_opener': make_variables_colorful_html(chat_gpt.memory_opener.template),
    }
    
    return render_template("chat.html",
                           history=formatted_history,
                           bot_name=BOT_NAME,
                           user_name=USER_NAME,
                           NEO4J_URI=NEO4J_URI.replace('neo4j+s','neo4j').replace('bolt+s','bolt'),
                           NEO4J_USERNAME=NEO4J_USERNAME,
                           NEO4J_PASSWORD=NEO4J_PASSWORD,
                           prompt_templates=prompt_templates,
                           llm_name = LLM_NAME,
                           lang = LANG
                           )



@app.route('/get')
def get_bot_response():
    user_input = request.args.get('msg')
    debug_mode = request.args.get('debug') == 'true'  # Get debug mode from the URL parameters
    anamnese_mode = request.args.get('anamnese') == 'true'  # Get debug mode from the URL parameters
    chat_gpt = ChatGPT(model=LLM_NAME, debug=debug_mode, anamnese_mode=anamnese_mode)
    chat_gpt.add_and_log_message("user", user_input)

    # Extract triplets
    # relationships = chat_gpt.extract_triplets()

    # Here, you could add your code to dump the relationships into the database, a file, or whatever you choose
    response = chat_gpt.generate_and_add_response()
    return str(response.content)

@app.route('/proactive')
def get_proactive_response():
    debug_mode = request.args.get('debug') == 'true'  # Get debug mode from the URL parameters
    anamnese_mode = request.args.get('anamnese') == 'true'  # Get debug mode from the URL parameters
    chat_gpt = ChatGPT(debug=debug_mode, anamnese_mode=anamnese_mode)

    
    # Generate a proactive message from the system (Adele)
    initial_prompt = chat_gpt.opener_generator.generate_opener()
    chat_gpt.add_and_log_message("system", initial_prompt)

    return str(initial_prompt)

@app.route('/proactive_memory')
def get_proactive_memory_response():
    debug_mode = request.args.get('debug') == 'true'  # Get debug mode from the URL parameters
    anamnese_mode = request.args.get('anamnese') == 'true'  # Get debug mode from the URL parameters
    chat_gpt = ChatGPT(model=LLM_NAME, debug=debug_mode, anamnese_mode=anamnese_mode)

    
    # Generate a proactive message from the system (Adele)
    opener, topic, strategy, relations, dialogue = chat_gpt.memory_opener.generate_opener()
    chat_gpt.add_and_log_message("system", opener)

    # Create a dictionary to hold the response data
    response_data = {
        'opener': opener,
        'topic': topic,
        'strategy': strategy,
        'relations': relations,
        'dialogue': dialogue
    }

    # Return the response data as JSON
    return json.dumps(response_data)

@app.route('/archive')
def archive_logs():
    debug_mode = request.args.get('debug') == 'true'  # Get debug mode from the URL parameters
    anamnese_mode = request.args.get('anamnese') == 'true'  # Get debug mode from the URL parameters
    chat_gpt = ChatGPT(model=LLM_NAME, debug=debug_mode, anamnese_mode=anamnese_mode)
    chat_gpt.dialogue_logger.archive_dialogue_logs()
    return "Logs have been archived successfully."

@app.route('/archives')
def list_archives():
    dialogue_logger = DialogueLogger(LOCAL_RAW_DATA_PATH / 'dialogue_logs')
    archive_folders = dialogue_logger.get_archive_folders()
    return json.dumps(archive_folders)

@app.route('/load_archive/<archive_name>')
def load_archive(archive_name):
    debug_mode = request.args.get('debug') == 'true'  # Get debug mode from the URL parameters
    anamnese_mode = request.args.get('anamnese') == 'true'  # Get debug mode from the URL parameters
    chat_gpt = ChatGPT(model=LLM_NAME, debug=debug_mode, anamnese_mode=anamnese_mode)
    chat_gpt.reload_from_archive(archive_name)
    return f"Archive '{archive_name}' loaded successfully."

@app.route('/is_graph_empty')
def is_graph_empty():
    graph = Neo4jGraph()
    result = graph.is_graph_empty()
    return jsonify({'is_empty': result})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080) 
