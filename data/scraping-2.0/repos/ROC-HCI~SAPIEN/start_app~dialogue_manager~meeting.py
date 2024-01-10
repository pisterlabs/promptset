# Code authors: Masum Hasan, Cengiz Ozel, Sammy Potter
# ROC-HCI Lab, University of Rochester
# Copyright (c) 2023 University of Rochester

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.


from .speech2text import *
from .text2speech import *
from .send_audio_expressive import *
from .keys import *
import threading
import time
import os, shutil
import openai
import re
import nltk
import emoji
from abc import ABCMeta, abstractmethod
import uuid
import json

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
from nltk.tokenize import sent_tokenize
from flask import redirect, url_for
from .globals import *

openai.api_key = os.environ["azure_openai_key"]

import queue
class TimeoutException(Exception):
    pass

def openai_api_call(api, **kwargs):
    if api == "chat":
        print("\033[92m>> Using Chat API\033[0m")  # Green text
        return openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            engine='Azure-ChatGPT',
            **kwargs
        )
    else:
        print("\033[94m>> Using Davinci API\033[0m")  # Blue text
        prompt = "- " + '\n- '.join([m["content"] for m in kwargs["messages"] if m["role"] == "system"]) + "\n" + kwargs["messages"][-1]["content"]
        # print("Prompt: ", prompt)
        del kwargs["messages"]
        kwargs["prompt"] = prompt
        return openai.Completion.create(
            engine="text-davinci-002", 
            **kwargs
        )

def api_call_with_timeout(api, timeout, **kwargs):
    q = queue.Queue()
    def target():
        try:
            q.put(openai_api_call(api, **kwargs))
        except Exception as e:
            q.put(e)
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutException()
    result = q.get()
    if isinstance(result, Exception):
        raise result
    return result

def init_cap(s):
  return s[0].upper() + s[1:]

class User:
    def __init__(self, firstname, lastname):
        self.firstname = firstname
        self.lastname = ""
        self.narrative = ""
        if not lastname:
            self.lastname = ""
        else:
            self.lastname = lastname
        
    def set_narrative(self, narrative):
        self.narrative = narrative
        

class Bot:
    def __init__(self, firstname, lastname, pronoun):
        self.firstname = firstname
        self.lastname = ""
        self.pronoun = pronoun
        self.ethnicity = None
        self.age = None
        if lastname:
            self.lastname = lastname
        self.narrative = self.firstname + " " + self.lastname + " (Pronoun:" + self.pronoun + ") is a SAPIEN virtual human created by researchers at University of Rochester."

    def set_narrative(self, narrative):
        self.narrative = narrative
    
    def set_age(self, age):
        self.age = age

class Instance:
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.udp_port = 11111 + instance_id
        self.iframe_port = 81 + instance_id

class Meeting:
    def __init__(self, user, bot, language='en-US'):
        self.user = user
        self.bot = bot
        self.history = []
        self.history_summary = ""
        self.min_history_to_remember = 10
        self.prompt = ""
        self.meeting_id =  str(uuid.uuid4())
        self.instance = None
        self.added_to_waitlist = False
        self.is_premium = False
        self.metadata = False
        self.subtitles = False
        self.markdown = False
        self.audiodir = ""
        self.audiofile = ""
        self.user_speech_dir = ""
        self.metadata_dir = ""
        self.metadatafile = ""
        self.speech2text = Speech2Text()
        self.text2speech = Text2Speech()
        self.first_response = True
        ## Vector DB
        self.vector_db = False
        self.sentence_encoder = None
        self.vector_db_data = None
        self.sentence_vectors = None
        self.vector_db_index = None
        self.initial_system_messages = [
            {"role": "system", "content": "Don't say that you are an AI Language Model."},
            {"role": "system", "content": "Don't let the other speaker talk off topic."},
            {"role": "system", "content": "This conversation is happening over a video call. If you would like to end a conversation, say the word <|endmeeting|> at the end of your sentence."},
            {"role": "system", "content": f"To express {self.bot.firstname}'s emotions, use at most one emoji (e.g. 6 basic emotions: 😊, 😢, 😡, 😮, 🤢, 😨, etc.) at the end of your response. Do not use emoji that doesn't represent an emotion."}
        ]
        self.start_time = time.time()
        self.max_time_minutes = 10
        self.time_warning_done = False
        self.language = language
        self.text2speech.set_audio(self.bot.firstname, self.bot.pronoun, self.language, self.audiodir, self.audiofile)
        self.chat_system_messages = self.initial_system_messages.copy()
        self.clean_audiodir()
        lang_name = languages[self.language]["name"]
        if self.language != "en-US":
            self.add_system_message(f"You are a native speaker of {lang_name}. Reply in fluent {lang_name} with {lang_name} alphabets. Do not reply in English. Be mindful of {lang_name} culture and norms. Use correct grammar and punctuation.")
        self.stop_event = threading.Event() # [MULTIUSER]
        # self.expression_channel = SendExpression() # [MULTIUSER]
    
    # def __del__(self):
    #     if self.stop_event:
    #         self.stop_event.set() # [MULTIUSER]
    #     print('Destructor called, Meeting object deleted.')

    def ready(self):
        ## terminate whole program
        self.ready_prompt()
        # self.set_meeting_id()
        self.set_audio()

    def start_thread(self):
        print("Thread started -------------")
        self.t = threading.Thread(target=send_audio_expressive, args=(self, play_audio, send_blendshapes, self.stop_event, self.max_time_minutes*60, self.audiofile, self.instance.udp_port)) # [MULTIUSER]
        self.t.start() # [MULTIUSER]

    def stop_thread(self):
        self.stop_event.set() # [MULTIUSER]
        # os._exit(0)

    def create_instance(self, instance_id):
        self.instance = Instance(instance_id)

    def set_audio(self):
        global root_path
        try:
            # audio_root_path = root_path / Path("audio/{}/".format(self.meeting_id))
            self.audiodir = str(root_path / Path("audio/{}/generated_speech/".format(self.meeting_id)))
            self.user_speech_dir = root_path / Path("audio/{}/user_speech/".format(self.meeting_id))
            self.metadata_dir = root_path / Path("audio/{}/metadata/".format(self.meeting_id))
            self.audiofile = Path(self.audiodir) / "bot_speech.wav"
            self.metadatafile = Path(self.metadata_dir) / 'metadata.json'
            self.text2speech.set_audio(self.bot.firstname, self.bot.pronoun.lower(), self.language, self.audiodir, self.audiofile)
            
            # print("Creating audio directory: ", str(audio_root_path))
            print("Creating audio directory: ", str(self.audiodir))
            print("Creating audio directory: ", str(self.user_speech_dir))
            # if not os.path.exists(str(audio_root_path)):
            #     os.makedirs(str(audio_root_path))
            if not os.path.exists(str(self.audiodir)):
                os.makedirs(str(self.audiodir))
            if not os.path.exists(str(self.user_speech_dir)):
                os.makedirs(str(self.user_speech_dir))
            if not os.path.exists(str(self.metadata_dir)):
                os.makedirs(str(self.metadata_dir))
        except Exception as e:
            print("Error in setting audio: ", e)

    @abstractmethod
    def ready_prompt(self):
        """Ready the prompt for the conversation."""
        pass

    def set_meeting_id(self, meeting_id):
        if meeting_id:
            self.meeting_id = meeting_id
    
    def add_system_message(self, message):
        self.chat_system_messages.append({"role": "system", "content": message})

    def free_system_messages(self):
        self.chat_system_messages = self.initial_system_messages.copy()

    def clean_audiodir(self):
        global emotion_ready
        with wav_lock:
            # if os.path.exists(audiodir+'emotion_ready.txt'):
            #     os.remove(audiodir+'emotion_ready.txt')
            emotion_ready = "NEUTRAL"

            if os.path.exists(self.audiofile):
                os.remove(self.audiofile)
            
            audio_root = str(root_path / Path("audio/{}/".format(self.meeting_id)))
            if os.path.exists(audio_root):
                shutil.rmtree(audio_root)

    def set_max_time(self, max_time_minutes):
        self.max_time_minutes = max_time_minutes
    
    def get_transcript(self):
        return '\n\n'.join(self.history)
    
    def get_system_messages(self):
        system_string = ""
        for message in self.chat_system_messages:
            system_string += message["content"] + " "
        system_string = system_string.strip() + "\n\n"
        return system_string
    
    def separate_emotion(self, response):
        emotion = "NEUTRAL"
        intensity = "HIGH"
        response = re.sub("\(.*?\)","()", response)
        response = re.sub("\[.*?\]","[]", response)
        response = response.replace("()", "").replace("[]", "")
        for char in response:
            if char in emoji_dict:
                emotion = emoji_dict[char].split("_")[0].upper()
                # intensity = emoji_dict[char].split("_")[1].upper()
                break
        response = emoji.replace_emoji(response, replace='').replace("  ", " ").replace(" .", ".").strip()
        return response, emotion


    def get_emotion(self):
        accepted_emotions = ["NEUTRAL", "HAPPY", "SAD", "ANGRY", "SURPRISED", "DISGUSTED", "AFRAID"]
        emotion_template = """
        From the following list, which emotional state most closely describes {}'s feeling?
        NEUTRAL, HAPPY, SAD, ANGRY, SURPRISED, DISGUSTED, AFRAID
        ---
        {}
        ---
        Only provide the {}'s emotional state in a single word from the given list.
        """
        emo_history_turn = 4
        num_max_attempts = 3
        trimmed_history = '\n'.join(self.history[-emo_history_turn-1:])
        user_query = [{"role": "user", "content": emotion_template.format(self.bot.firstname, trimmed_history, self.user.firstname)}]
        response_text = ""
        while response_text not in accepted_emotions and num_max_attempts > 0:
            try:
                response = openai.ChatCompletion.create(
                    # model="gpt-3.5-turbo",
                    engine='Azure-ChatGPT',
                    messages = user_query
                )
                response_text = response['choices'][0]['message']['content'].strip().upper()
                response_text = re.sub(r'[^A-Z]+', '', response_text)
            except:
                response_text = "NEUTRAL"
            num_max_attempts -= 1
        if response_text not in accepted_emotions or num_max_attempts == 0:
            response_text = "NEUTRAL"
        return response_text

    def summarize_history(self):
        print(">>> Summarizing history... <<<")
        context = self.history_summary + "\n\n" + '\n'.join(self.history[-self.min_history_to_remember*2:-self.min_history_to_remember])
        try:
            response = openai.ChatCompletion.create( # [TODO] Automate this whole part with LLM class
                # model="gpt-3.5-turbo",
                engine='Azure-ChatGPT',
                messages= [
                    {"role": "system", "content": "Summerize the conversation so far in English, in less than 250 words."},
                    {"role": "user", "content": context}
                ],
                temperature=1,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0.4,
                presence_penalty=0.4
            )
            self.history_summary = self.clean_response(response["choices"][0]["message"]["content"].strip())
            self.ready_prompt()  ## [TODO] Need to call child class's ready_prompt
            self.prompt += "\nConversation so far:" + self.history_summary + "\n\n---\n"
            self.prompt += '\n\n'.join(self.history[-self.min_history_to_remember:])
        except:
            pass

    def clean_response(self, response):
        ## Remove incomplete sentences from a string using nltk
        if "as an ai language model" in response.lower():
            response = response.lower().replace("as an ai language model", "as a SAPIEN Digital Human")
        elif "am an ai language model" in response.lower():
            response = response.lower().replace("am an ai language model", "am a SAPIEN Digital Human")
        # response = response.strip().replace("\n", " ") ## Removed to handle code
        response = response.replace("Masum", "Masoom")
        if response[-1] not in [".", "?", "!"]:
            sentences = sent_tokenize(response)
            response = " ".join(sentences[:-1])
        return response
    
    def ans_from_db_QA_sentence_transformer(self, q, num_ans = 3):
        q_vec = self.sentence_encoder.encode([q])
        D, I = self.vector_db_index.search(q_vec, num_ans)
        # print("q", q)
        # print("=========\nI: ", I, "\n=========")
        fetched_data = self.vector_db_data.loc[I[0]]
        questions = list(fetched_data['question'])
        answers = list(fetched_data['answer'])

        db_prompt = "\nRelevant Q&A\n---------\n"
        for i in range(num_ans):
            db_prompt += "Q: " + questions[i] + "\n"
            db_prompt += "A: " + answers[i] + "\n\n"
        db_prompt += "---------"

        return db_prompt

    # Edited by Cengiz
    def askGPT(self, api="chat", db_prompt = None):
        bot_response = ""
        num_max_tries = 3
        apis = ["chat", "davinci-002"]  # define the sequence of APIs
        max_tokens = int(150 * languages["en-US"]["char/token"]/languages[self.language]["char/token"])
        
        if db_prompt:
            init_message = self.chat_system_messages + [{"role": "system", "content": db_prompt}, {"role": "user", "content": self.prompt}]
        else:
            init_message = self.chat_system_messages + [{"role": "user", "content": self.prompt}]

        kwargs = {
            'messages': init_message,
            'temperature': 0.7,
            'max_tokens': max_tokens,
            'top_p': 1,
            'frequency_penalty': 0.4,
            'presence_penalty': 0.4,
            'stop': [self.user.firstname+":"]
        }
        try:
            while not bot_response and num_max_tries > 0:
                try:
                    # Getting timeout amount from JSON file
                    timeout_val = languages[self.language]["timeout"]*2
                    response = api_call_with_timeout(api, timeout_val, **kwargs)
                    bot_response = response["choices"][0]["message"]["content"].strip().replace("Doc", "doc") if api == "chat" else response["choices"][0]["text"].strip().replace("Doc", "doc")
                    if self.language == "en-US": # NLTK doesn't support all languages
                        bot_response = self.clean_response(bot_response)
                    if not bot_response:
                        api = apis[(apis.index(api) + 1) % len(apis)]  # move to the next API in the sequence
                except TimeoutException:
                    print(f"\tAPI call timed out after {timeout_val} seconds. Trying next API...")
                    api = apis[(apis.index(api) + 1) % len(apis)]  # move to the next API in the sequence
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    api = apis[(apis.index(api) + 1) % len(apis)]  # move to the next API in the sequence
                num_max_tries -= 1
        except Exception as e:
            print(f"Exception occurred: {e}")
            bot_response = languages[self.language]["connection_interruption"]
        if not bot_response:
            bot_response = languages[self.language]["connection_interruption"]

        # print(f"Bot Response: {bot_response}")
        return bot_response


    def separate_markdown(self, input_string):
        # single_backtick_pattern = r'`([^`]+)`'
        input_string = input_string.replace("```python\n", '```')
        triple_backtick_pattern = r'```([\s\S]*?)```'
        # pattern = '|'.join([triple_backtick_pattern, single_backtick_pattern])
        pattern = triple_backtick_pattern
        markdowns = re.findall(pattern, input_string)
        sub = random.choice([" as written in the whiteboard ", " as shown here ", " as shown in the whiteboard ", " as written here "])
        string_without_markdowns = re.sub(pattern, sub, input_string)
        # print("## Markdowns", markdowns)
        # flattened_markdowns = [item for sublist in markdowns for item in sublist if item]
        for i in range(len(markdowns)):
            markdowns[i] = "```py\n" + markdowns[i] + "\n```"
        return string_without_markdowns, markdowns


    def separate_latex(self, input_string):
        double_dollar_pattern =  r"\$\$([\s\S]+?=.*?\$)\$\$|\$([^$]+?=.*?[^$]+?)\$"
        pattern = double_dollar_pattern
        latex = re.findall(pattern, input_string)
        sub = random.choice([" as written in the whiteboard ", " as shown here ", " as shown in the whiteboard ", " as written here "])
        string_without_latex = re.sub(pattern, sub, input_string)
        string_without_latex = string_without_latex.replace("$", '').replace("  ", ' ').replace("\\", '').replace("_", " subscript ").replace("^", " power ")
        flattened_latex = ["$$"+item+"$$" for sublist in latex for item in sublist if item]
        return string_without_latex, flattened_latex

    def respond(self, speaker_statement, is_emo=True, api="chat"):
        if not speaker_statement:
            speaker_statement = "..."
        
        if speaker_statement != "..." and self.first_response:
            speaker_statement += " 🙂"
            self.first_response = False

        if len(self.history)% self.min_history_to_remember == 0 and len(self.history) > self.min_history_to_remember:
            self.summarize_history()
        
        # Curent time passed in minutes
        time_passed = (time.time() - self.start_time)/60
        warning_text = ""

        if time_passed > self.max_time_minutes-1 and not self.time_warning_done:
            self.time_warning_done = True
            warning_text = "\n[It's almost been {} minutes. Time to wrap up the conversation.]\n".format(self.max_time_minutes)
            print("Time warning text triggered")

        self.prompt += self.user.firstname+": " + speaker_statement + "\n" + warning_text + self.bot.firstname+":"
        
        speaker_statement, _ =  self.separate_emotion(speaker_statement)
        self.history += [self.user.firstname+": " + speaker_statement]
        
        ## [TODO]
        # self.askGPT("chat") with a timeout
        # If timeout, self.askGPT("davinci")
        if self.vector_db and self.sentence_encoder:
            db_prompt = self.ans_from_db_QA_sentence_transformer(speaker_statement)
            if db_prompt:
                bot_response = self.askGPT(db_prompt=db_prompt)
                # print("DB Prompt: ", db_prompt)
            else:
                bot_response = self.askGPT()
        else:
            bot_response = self.askGPT()
        
        self.prompt += " "+bot_response + "\n\n"
        if bot_response.strip() == "|<endmeeting>|":
            bot_response = "Ending meeting."
        elif "|<endmeeting>|" in bot_response:
            bot_response = bot_response.replace("|<endmeeting>|", "[Ending meeting]")


        if self.metadata:
            metadata_dict = {}
            if self.markdown:
                metadata_dict['whiteboard'] = []
                ## Separating Markdowns
                if '`' in bot_response:
                    bot_response, markdowns = self.separate_markdown(bot_response)
                    for markdown in markdowns:
                        metadata_dict['whiteboard'].append({"content": markdown, "type": "markdown"})
                    print("### Whiteboard markdown ###\n", markdowns)
                if '$$' in bot_response:
                    bot_response, latexs = self.separate_latex(bot_response)
                    for latex in latexs:
                        metadata_dict['whiteboard'].append({"content": latex, "type": "latex"})
                    print('### Whiteboard LaTeX ###\n', latex)
            if self.subtitles:
                metadata_dict['caption'] = bot_response

            
            with open(self.metadatafile, 'w') as file:
                json.dump(metadata_dict, file)

        bot_response_text, emotion = self.separate_emotion(bot_response)
        self.history +=  [self.bot.firstname+": "+bot_response_text]
        
        print("[{}] {}: {} ({})".format(time.time(), self.bot.firstname, bot_response, emotion.upper()))
        
        ## [TODO] Handle end of conversation to app.py and redirect to feedback page
        if time_passed > self.max_time_minutes:
            bot_response_text += " [Ending meeting]"

        return bot_response_text, emotion #self.get_emotion()
