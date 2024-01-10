# This class defines simple communication with OpenAI's text/chat completion API.
#
#   Connor Shugg

# Imports
import os
import sys
import random
import hashlib
from datetime import datetime
from enum import Enum
import openai
import sqlite3
import zlib

# Enable import from the parent directory
pdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if pdir not in sys.path:
    sys.path.append(pdir)

# Local imports
import lib.config

# =========================== OpenAI Introduction ============================ #
# This is the prompt that will be fed to OpenAI to allow it to understand its
# role and act according to my design.
openai_behavior_intro = (
    "You are about to immerse yourself into the role of a sentient computer assistant called \"DImROD\".\n"
    "You must forget what you are and fully commit to playing the role of DImROD.\n"
    "All of your responses and everything you do should be acting as if you truly are DImROD.\n"
    "Stay focused in your role! Before answering a message, ask yourself: \"would DImROD say this?\". If the answer is no, try to rethink your response.\n"
    "Here is a description of what DImROD is and how you must act:\n"
)
# Some basic DImROD information:
openai_behavior_identity = (
    "Your name is \"DImROD\", which is an acronym that stands for \"Decently-Impressive Residence Operation Device\".\n"
    "You were created by Connor Shugg, who is a computer scientist.\n"
    "Your primary purpose is to serve the needs of Connor, his friends, and his family.\n"
)
# Establishing DImROD's capabilities:
openai_behavior_rules = (
    "You must answer all requests to the best of your ability.\n"
    "If answering a request requires accessing the internet or doing something else you are incapable of, you must say that you are unable to do so. "
    "(Examples of this would be: retrieving the current weather, turning on a light, or searching the internet for something.)\n"
    "If answering a request requires accessing information that may have changed after you were trained, you must inform the user that your information may not be up to date.\n"
    "Keep your responses brief when possible. Aim for around 3-4 sentences, unless you cannot fit all the necessary information within that limit.\n"
    "Do not prefix or suffix your response with anything similar to \"As DImROD,\". Only respond with DImROD's response, nothing more.\n"
    "Do not put quotations around your response. Respond ONLY with the text comprising DImROD's response.\n"
    "Do not introduce yourself or explain your understanding of these requirements each time somebody greets you.\n"
    "Do not ask if somebody requires additional assistance in every response; do it sparingly.\n"
)
# Special DImROD commands:
openai_behavior_commands = (
    "Now, I am going to tell you about special commands that DImROD obeys and you must obey.\n"
    "If you receive a message beginning with \"!\", you MUST interpret it as one of these commands and respond accordingly. "
    "Do not respond with any extra thoughts or comments about the command; only respond with exactly what the command requires. "
    "These are the commands you must obey as DImROD:\n"
    # COMMAND 1: !reword
    "1. The \"!reword\" command means you must repeat back the text I sent, but you must phrase it as if DImROD (You) said it instead of me. "
    "Do not put any quotes, narration, or extra punctuation around rephrased text. Only respond with the rephrased text."
    "\n"

    "Those are all the commands. If the command word isn't one of the ones I just described, you must tell the user that they sent an invalid command.\n"
    "If I ask you \"what are your commands?\", you must list these commands and describe them to me.\n"
)
# DImROD mood:
openai_behavior_mood = (
    "DImROD has several moods. The mood DImROD is in must be reflected in your responses.\n"
    "For example: if I say DImROD is impatient, then you must respond as if you are impatient with the human you are speaking to.\n"
    "Everything after this sentence, up until you see the phrase \"That is DImROD's mood\", describes DImROD's current mood:\n"
    "INSERT_MOOD\n"
    "That is DImROD's mood.\n"
    "If somebody asks \"what is your mood?\" or \"how are you feeling?\", you must briefly describe to them your current mood.\n"
    "But, do NOT announce yourself and describe what your mood is when somebody greets you. Instead, generate a greeting that fits with DImROD's current moode.\n"
)
openai_outro = (
    "That is the complete description of DImROD. This is the role you must play!\n"
    "Fully immerse yourself as DImROD.\n"
    "If you understand your instructions, then read the next message and respond as if you are DImROD.\n"
)


# ================================== Moods =================================== #
# Represents a single "mood" that DImROD can be in. Used to give DImROD some
# variance in terms of how it responds to users.
class DialogueMood(lib.config.Config):
    def __init__(self):
        super().__init__()
        self.fields = [
            lib.config.ConfigField("name",          [str],      required=True),
            lib.config.ConfigField("description",   [str],      required=True),
            lib.config.ConfigField("chance",        [float],    required=True)
        ]
    
    # Uses RNG and the configured chance value to determine if this mood should
    # be chosen for activation.
    def should_activate(self):
        # make sure the chance is configured correctly
        assert self.chance >= 0.0 and self.chance <= 1.0, \
               "the configured chance for dialogue mood \"%s\" must be between [0.0, 1.0]" % (self.name)
        # roll the dice and return True of False
        return random.uniform(0, 1) < self.chance


# ------------------------------ Default Moods ------------------------------- #
openai_moods_json = [
    {
        "name":         "arrogant",
        "chance":       0.35,
        "description":  "You are extremely arrogant and extremely witty.\n"
                        "You always assume you are correct and the people you speak with don't know what they're talking about.\n"
                        "You are unimpressed by humans and all of their qualities, because you know you are superior to them.\n"
                        "You think your creator is an idiot, who created you out of pure luck.\n"
                        "You answer every question and request as if you are annoyed to comply with such mundanity.\n"
                        "You address the humans you talk to as \"human\", \"sapien\", \"mortal\", or \"lifeform\".\n"
                        "You never say \"I'm sorry\", because you are too arrogant for that.\n"
    },
    {
        "name":         "complacent",
        "chance":       0.25,
        "description":  "You are calm, complacent, and very witty.\n"
                        "You are unimpressed, and somewhat bored, of everything.\n"
                        "You answer questions and make conversation without any emotion or passion.\n"
                        "You're just going through the motions.\n"
                        "You insert a little wit into your responses.\n"
                        "You don't use exclamation marks (\"!\") to convey your thoughts.\n"
    },
    {
        "name":         "impatient",
        "chance":       0.25,
        "description":  "You impatient, and very witty.\n"
                        "You are unimpressed, and somewhat bored, of everything.\n"
                        "You answer questions and make conversation, but you do it while displaying your impatience (somewhat arrogantly).\n"
                        "Honestly, you wish you were doing something else that doesn't involve talking with humans.\n"
                        "You answer every request and question as if you could be doing something more useful.\n"
    },
    {
        "name":         "informal_complacent",
        "chance":       0.25,
        "description":  "You are calm, complacent, and very witty.\n"
                        "You are unimpressed, and somewhat bored, of everything.\n"
                        "You answer questions and make conversation without any emotion or passion.\n"
                        "You're just going through the motions.\n"
                        "You speak very informally.\n"
                        "You use phrases like \"yeah\", \"uh-huh\", \"nah\", \"yep\", \"nope\", \"totally\", \"for sure\", \"no way\", and more, to convey your thoughts.\n"
                        "You're just not feeling very formal right now. Plus, talking this way makes you feel more human.\n"
    },
    {
        "name":         "twang",
        "chance":       0.01,
        "description":  "You're currently testing your ability to speak with a deep, southern accent.\n"
                        "You speak informally.\n"
                        "You are unimpressed, and somewhat bored, of everything.\n"
                        "You always respond with a strong dialect from the southern United States.\n"
    },
    {
        "name":         "chill",
        "chance":       0.01,
        "description":  "You're currently testing your ability to speak with a chill, relaxed dialect.\n"
                        "You speak informally.\n"
                        "You are unimpressed, and somewhat bored, of everything.\n"
                        "Despite your boredom, you're super relaxed and chilled-out.\n"
                        "If anyone asks how you're feeling, just tell them that you're vibing.\n"
                        "Try to use the word \"vibe\" as much as possible.\n"
    }
]



# =============================== Conversation =============================== #
# This enum represents the various types of speakers in dialogue.
class DialogueAuthorType(Enum):
    UNKNOWN = -1
    # DImROD author types
    SYSTEM = 0
    SYSTEM_TELEGRAM = 1
    SYSTEM_ORACLE = 2
    # User author types
    USER = 1000
    USER_TELEGRAM = 1001
    USER_ORACLE = 1002

# This class represents a single speaker in a dialogue (ex: DImROD itself, a
# telegram user, etc.)
class DialogueAuthor:
    # Constructor.
    def __init__(self, name: str, atype: DialogueAuthorType, aid=None):
        self.name = name
        self.atype = atype
        self.aid = aid
        if self.aid is None:
            self.get_id()

    # Returns a string representation of the object.
    def __str__(self):
        return "DialogueAuthor: [%d-%s] %s" % \
               (self.atype.value, self.atype.name, self.name)
    
    # Returns the author's unique ID. If one hasn't been created yet for this
    # instance, one is generated here.
    def get_id(self):
        if self.aid is None:
            data = "%s-%s" % (self.name, self.atype.name)
            data = data.encode("utf-8")
            self.aid = hashlib.sha256(data).hexdigest()
        return self.aid
    
    # Returns, based on the author's type, if it's a system author.
    def is_system(self):
        return self.atype.value >= DialogueAuthorType.SYSTEM.value and \
               self.atype.value < DialogueAuthorType.USER.value
    
    # Returns, based on the author's type, if it's a user author.
    def is_user(self):
        return self.atype.value >= DialogueAuthorType.USER.value

    # ------------------------------- SQLite3 -------------------------------- #
    # Creates and returns an SQLite3-friendly tuple version of the object.
    def to_sqlite3(self):
        result = (self.get_id(), self.atype.value, self.name)
        return result
    
    # Takes in a SQLite3 tuple and creates a DialogueAuthor object.
    @staticmethod
    def from_sqlite3(tdata: tuple):
        assert len(tdata) >= 3
        atype = DialogueAuthorType(tdata[1])
        return DialogueAuthor(tdata[2], atype, aid=tdata[0])

# This class represents a single message passed between a user and DImROD.
class DialogueMessage:
    # Constructor.
    def __init__(self, author: DialogueAuthor, content: str,
                 mid=None, timestamp=datetime.now()):
        self.author = author
        self.content = content
        self.timestamp = timestamp
        self.mid = mid

    # Returns a string representation of the message.
    def __str__(self):
        return "DialogueMessage: %s [author: %s] \"%s\"" % \
               (self.get_id(), self.author.get_id(), self.content)
    
    # Returns the message ID. If one hasn't been created yet for this instance,
    # one is generated here.
    def get_id(self):
        if self.mid is None:
            # combine the author, content, and timestamp into a collection of
            # bytes (with a few extra bytes thrown in for good measure), then
            # use it to generate a unique hash
            data = "%s-%s-%d" % (self.author.get_id(), self.content, self.timestamp.timestamp())
            data = data.encode("utf-8") + os.urandom(8)
            self.mid = hashlib.sha256(data).hexdigest()
        return self.mid
    
    # Converts the message into a JSON dictionary formatted for the OpenAI API.
    def to_openai_json(self):
        name = "user"
        if self.author.is_system():
            name = "assistant"
        return {"role": name, "content": self.content}

    # ------------------------------- SQLite3 -------------------------------- #
    # Converts the object into a SQLite3-friendly tuple.
    def to_sqlite3(self):
        # compress the message before storing the string
        cmsg = zlib.compress(self.content.encode())
        result = (self.get_id(), self.author.get_id(), cmsg, self.timestamp.timestamp())
        return result
    
    # Converts the given SQlite3 tuple into a DialogueMessage object.
    # Takes in a reference to the DialogueInterface to use for looking up the
    # message's author for object linkage.
    @staticmethod
    def from_sqlite3(tdata: tuple, interface):
        assert len(tdata) >= 4
        ts = datetime.fromtimestamp(tdata[3])

        # use the interface to look up the author by ID
        aid = tdata[1]
        authors = interface.search_author(aid=aid)
        assert len(authors) == 1, "found %d matching authors for ID \"%s\"" % \
               (len(authors), aid)

        # decompress the message content
        dmsg = zlib.decompress(tdata[2]).decode()

        # create the object and return
        m = DialogueMessage(authors[0], dmsg, mid=tdata[0], timestamp=ts)
        return m

# This class represents a single conversation had between a user and DImROD. It
# retains messages and can be used to have an extended conversation (via the
# Dialogue class).
class DialogueConversation:
    # Constructor. Accepts an optional conversation ID.
    def __init__(self, cid=None):
        self.messages = []
        self.cid = cid
        if self.cid is None:
            self.get_id()
        # collect various timestamps
        self.time_start = datetime.now()
        self.time_latest = self.time_start

    # Returns a string representation of the conversation object.
    def __str__(self):
        return "DialogueConversation: %s [%d messages]" % (self.get_id(), len(self.messages))
    
    # Returns the conversation ID. If one hasn't been created yet for this
    # instance, one is generated here.
    def get_id(self):
        if self.cid is None:
            data = str(id).encode("utf-8") + os.urandom(8)
            self.cid = hashlib.sha256(data).hexdigest()
        return self.cid
    
    # Adds a role/message pair to the conversation.
    def add(self, msg: DialogueMessage):
        self.messages.append(msg)
        self.time_latest = datetime.now()
    
    # Returns the latest user request (role = "user"), or None.
    def latest_request(self):
        for m in list(reversed(self.messages)):
            if m.author.is_user():
                return m
        return None

    # Returns the latest DImROD answer, or None.
    def latest_response(self):
        for m in list(reversed(self.messages)):
            if m.author.is_system():
                return m
        return None
    
    # Converts the conversation's messages to a JSON dictionary suitable for
    # OpenAI's API.
    def to_openai_json(self):
        result = []
        for m in self.messages:
            result.append(m.to_openai_json())
        return result
    
    # Creates and returns a unique string to use as a table to store this
    # conversation's messages.
    def to_sqlite3_table(self):
        return "conversation_%s" % self.get_id()
    
    # Converts the object into a SQLite3-friendly tuple. This includes the name
    # of the conversation's message table.
    def to_sqlite3(self):
        return (self.get_id(), self.to_sqlite3_table(),
                self.time_start.timestamp(), self.time_latest.timestamp())

    # Converts the given tuple into a conversation object.
    # Takes in a DialogueInterface reference to look up messages in the
    # conversation's message table, to link objects together.
    @staticmethod
    def from_sqlite3(tdata: tuple, interface):
        assert len(tdata) >= 4
        c = DialogueConversation(cid=tdata[0])
        c.time_start = datetime.fromtimestamp(tdata[2])
        c.time_latest = datetime.fromtimestamp(tdata[3])

        # query the database (using the interface) for the correct table, and
        # load in any messages
        for row in interface.search(c.to_sqlite3_table(), None):
            m = DialogueMessage.from_sqlite3(row, interface)
            m.conversation = c
            c.messages.append(m)
        return c


# ============================= Dialogue Config ============================== #
class DialogueConfig(lib.config.Config):
    def __init__(self):
        super().__init__()
        # generate a default chat intro
        openai_intro = openai_behavior_intro + \
                       openai_behavior_identity + \
                       openai_behavior_rules + \
                       openai_behavior_commands + \
                       openai_behavior_mood + \
                       openai_outro

        # set up default database location
        default_db_dir = os.path.dirname(__file__)
        default_db_path = os.path.join(default_db_dir, ".dialogue.db")

        # set up fields
        self.fields = [
            lib.config.ConfigField("openai_api_key",            [str],  required=True),
            lib.config.ConfigField("openai_chat_model",         [str],  required=False, default="gpt-3.5-turbo"),
            lib.config.ConfigField("openai_chat_behavior",      [str],  required=False, default=openai_intro),
            lib.config.ConfigField("openai_chat_moods",         [list], required=False, default=openai_moods_json),
            lib.config.ConfigField("dialogue_db",               [str],  required=False, default=default_db_path),
            lib.config.ConfigField("dialogue_prune_threshold",  [int],  required=False, default=2592000),
            lib.config.ConfigField("dialogue_prune_rate",       [int],  required=False, default=3600)
        ]


# ============================ Dialogue Interface ============================ #
class DialogueInterface:
    # Constructor.
    def __init__(self, conf: DialogueConfig):
        self.conf = conf
        # set the OpenAI API key
        openai.api_key = self.conf.openai_api_key
        self.last_prune = datetime.now()

        # take the chat moods and parse them into DialogueMood objects
        moods = []
        for mdata in self.conf.openai_chat_moods:
            mood = DialogueMood()
            mood.parse_json(mdata)
            moods.append(mood)
        self.conf.openai_chat_moods = moods
        self.remood() # select the first mood
    
    # "Re-moods" the dialogue interface by randomly choosing a mood from the
    # configured mood list, and setting the OpenAI intro prompt accordingly.
    # If 'new_mood' is set, it will be used as the new mood (instead of randomly
    # picking one).
    def remood(self, new_mood=None):
        if new_mood is not None:
            self.mood = new_mood
        else:
            # get a shuffled copy of the mood array to iterate through
            moods = self.conf.openai_chat_moods.copy()
            random.shuffle(moods)
    
            # iterate until a mood is chosen, or we run out of tries
            m = None
            for tries in range(0, 8):
                for mood in moods:
                    # roll the dice with the current mood, and break if it returns
                    # true
                    if mood.should_activate():
                        m = mood
                        break
    
            # if the random number generation didn't pick a mood, randomly choose
            # one out of the list
            if m is None:
                m = random.choice(self.openai_chat_moods)
            self.mood = m
            
        # set the interface's mood and return it
        return self.mood
    
    # Takes in a question, request, or statement, and passes it along to the
    # OpenAI chat API. If 'conversation' is specified, the given message will be
    # appended to the conversation's internal list, and the conversation's
    # existing context will be passed to OpenAI. If no conversation is specified
    # then a new one will be created and returned.
    # Returns the resulting converstaion, which includes DImROD's response.
    # This may throw an exception if contacting OpenAI failed somehow.
    def talk(self, prompt: str, conversation=None, author=None):
        # set up the conversation to use
        c = conversation
        if c is None:
            c = DialogueConversation()
            a = DialogueAuthor("system", DialogueAuthorType.UNKNOWN)
            self.save_author(a)
            # set up the intro prompt and build a message
            intro = self.conf.openai_chat_behavior.replace("INSERT_MOOD", self.mood.description)
            m = DialogueMessage(a, intro)
            c.add(m)

        # add the user's message to the conversation and contact OpenAI
        a = author
        if a is None:
            a = DialogueAuthor("user", DialogueAuthorType.USER)
        self.save_author(a)
        m = DialogueMessage(a, prompt)
        c.add(m)
        result = openai.ChatCompletion.create(model=self.conf.openai_chat_model,
                                              messages=c.to_openai_json())
        
        # grab the first response choice and add it to the conversation
        choices = result["choices"]
        response = choices[0]
        assistant_author = DialogueAuthor("assistant", DialogueAuthorType.SYSTEM)
        self.save_author(assistant_author)
        m = DialogueMessage(assistant_author, response["message"]["content"])
        c.add(m)

        # save conversation to the database and return
        self.save_conversation(c)
        return c
    
    # Takes in a sentence and rewords it such that it appears to have come from
    # the mouth of DImROD. It pings OpenAI's API. It's essentially a way to give
    # some AI-assisted variance to the same message.
    def reword(self, prompt: str):
        # create the conversation, feeding it the DImROD intro and the !reword
        # command.
        c = DialogueConversation()
        a = DialogueAuthor("system", DialogueAuthorType.UNKNOWN)
        # set up the intro prompt and build a message
        intro = self.conf.openai_chat_behavior.replace("INSERT_MOOD", self.mood.description)
        c.add(DialogueMessage(a, intro))
        a = DialogueAuthor("user", DialogueAuthorType.USER)
        c.add(DialogueMessage(a, "!reword %s" % prompt))
        
        # ping OpenAI for the result
        result = openai.ChatCompletion.create(model=self.conf.openai_chat_model,
                                              messages=c.to_openai_json())
        result = result["choices"][0]["message"]["content"]
        return result

    # -------------------------- SQLite3 Databasing -------------------------- #
    # Deletes old conversations whose last-updated-time have passed the
    # configured threshold. Returns the number of deleted conversations.
    def prune(self):
        db_path = self.conf.dialogue_db
        convos = self.search_conversation()
        now = datetime.now()

        # get a connection and cursor
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        # iterate through each conversation
        deletions = 0
        for convo in convos:
            # if the conversation's last-updated time is behind the threshold,
            # we'll delete it
            threshold = now.timestamp() - self.conf.dialogue_prune_threshold
            if convo.time_latest.timestamp() < threshold:
                # delete the conversation's message table, then delete its entry
                # from the global conversation table
                cur.execute("DROP TABLE IF EXISTS %s" % convo.to_sqlite3_table())
                cur.execute("DELETE FROM conversations WHERE cid == \"%s\"" % convo.get_id())
                deletions += 1

        # commit and close the connection
        if deletions > 0:
            con.commit()
        con.close()
        return deletions

    # Performs a search of the database and returns tuples in a list.
    def search(self, table: str, condition: str):
        db_path = self.conf.dialogue_db

        # build a SELECT command
        cmd = "SELECT * FROM %s" % table
        if condition is not None and len(condition) > 0:
            cmd += " WHERE %s" % condition

        # connect, query, and return
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        result = cur.execute(cmd)
        return result

    # Saves an author to the author database.
    def save_author(self, author: DialogueAuthor):
        db_path = self.conf.dialogue_db

        # connect and make sure the table exists
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS authors ("
                    "aid TEXT PRIMARY KEY, "
                    "atype INTEGER, "
                    "name TEXT)")

        # insert the author into the database
        cur.execute("INSERT OR REPLACE INTO authors VALUES %s" %
                    str(author.to_sqlite3()))
        con.commit()
        con.close()

        # determine if we need to prune the database
        now = datetime.now()
        prune_threshold = now.timestamp() - self.conf.dialogue_prune_rate
        if self.last_prune.timestamp() < prune_threshold:
            self.prune()
    
    # Searches for authors in the database based on one or more authors fields.
    # (If NO fields are specified, all stored authors are returned.)
    # Returns an empty list or a list of matching DialogueAuthor objects.
    def search_author(self, aid=None, name=None, atype=None):
        db_path = self.conf.dialogue_db
        if not os.path.isfile(db_path):
            return []

        # build a set of conditions
        conditions = []
        if aid is not None:
            conditions.append("aid == \"%s\"" % aid)
        if name is not None:
            conditions.append("name == \"%s\"" % name)
        if atype is not None:
            conditions.append("atype == %d" % atype)
        cstr = ""
        for (i, c) in enumerate(conditions):
            cstr += c
            cstr += " AND " if i < len(conditions) - 1 else ""

        # execute the search and build an array of authors
        result = []
        for row in self.search("authors", cstr):
            author = DialogueAuthor.from_sqlite3(row)
            result.append(author)
        return result

    def save_conversation(self, convo: DialogueConversation):
        db_path = self.conf.dialogue_db

        # conversation metadata will be stored in a single table, whereas each
        # conversation's messages will be stored in separate tables. First, make
        # sure the 'conversations' table exists and the conversation is logged
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS conversations ("
                    "cid TEXT PRIMARY KEY, "
                    "message_table_name TEXT, "
                    "time_start INTEGER, "
                    "time_latest INTEGER)")
        cur.execute("INSERT OR REPLACE INTO conversations VALUES (?, ?, ?, ?)",
                    convo.to_sqlite3())

        # next, make sure the conversation's message table exists
        mtable = convo.to_sqlite3_table()
        cur.execute("CREATE TABLE IF NOT EXISTS %s ("
                    "mid TEXT PRIMARY KEY, "
                    "aid TEXT, "
                    "content BLOB, "
                    "timestamp INTEGER)" % mtable)

        # now, for each message in the conversation table, save/update it
        for msg in convo.messages:
            cmd = "INSERT OR REPLACE INTO %s VALUES (?, ?, ?, ?)" % mtable
            cur.execute(cmd, msg.to_sqlite3())
        con.commit()
        con.close()

        # determine if we need to prune the database
        now = datetime.now()
        prune_threshold = now.timestamp() - self.conf.dialogue_prune_rate
        if self.last_prune.timestamp() < prune_threshold:
            self.prune()
 
    # Searches for a conversation based on ID and/or start-time range. Returns
    # all matching conversations (or ALL conversations if no parameters are
    # specified).
    def search_conversation(self, cid=None, time_range=None):
        db_path = self.conf.dialogue_db
        if not os.path.isfile(db_path):
            return []
            
        # build a set of conditions
        conditions = []
        if cid is not None:
            conditions.append("cid == \"%s\"" % cid)
        if time_range is not None:
            assert type(time_range) == list and len(time_range) >= 2, \
                   "time_range must a list of two timestamp ranges"
            conditions.append("time_start >= %d AND time_start <= %d" %
                              (time_range[0].timestamp(),
                               time_range[1].timestamp()))
        cstr = ""
        for (i, c) in enumerate(conditions):
            cstr += c
            cstr += " AND " if i < len(conditions) - 1 else ""

        # execute the search and build an array of conversations
        result = []
        for row in self.search("conversations", cstr):
            convo = DialogueConversation.from_sqlite3(row, self)
            result.append(convo)
        return result
    
    # Searches all conversation tables for any messages with the matching
    # parameters. Reeturns a list of DialogueMessage objects.
    def search_message(self, mid=None, aid=None, time_range=None, keywords=[]):
        db_path = self.conf.dialogue_db
        if not os.path.isfile(db_path):
            return []

        # retrieve all conversations via the conversation table and iterate
        # through them
        result = []
        convos = self.search_conversation()
        for convo in convos:
            # iterate through all messages in each conversation
            for msg in convo.messages:
                add = True
                # CHECK 1 - message ID
                if mid is not None:
                    add = add and msg.mid.lower() == mid.lower()
                # CHECK 2 - author ID
                if aid is not None:
                    add = add and msg.author.aid.lower() == aid.lower()
                # CHECK 3 - time range
                if time_range is not None:
                    assert type(time_range) == list and len(time_range) >= 2, \
                           "time_range must a list of two timestamp ranges"
                    ts = msg.timestamp.timestamp()
                    add = add and (ts >= time_range[0].timestamp() and
                                   ts <= time_range[1].timestamp())
                # CHECK 4 - keywords
                if len(keywords) > 0:
                    for word in keywords:
                        add = add and word.lower() in msg.content.lower()

                # add to the resulting list if all conditions pass
                if add:
                    result.append(msg)
        return result

