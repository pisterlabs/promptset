

#BLACKJACK DATABASE BY PRIMAL @101KEK
#FEEL FREE TO SKID MY BOT PLEASE GIVE CREDIT.

import logging
import sqlite3
import threading

class BlackjackDatabase:
    def __init__(self, db_path, table_suffix):
        self.db_path = db_path
        self.table_suffix = table_suffix
        self.lock = threading.Lock()
        self.setup_database()
        self.user_data = {}  # Dictionary to store user data

    def create_connection(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            return conn
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")
            return None

    def setup_database(self):
        with self.lock:
            try:
                with self.create_connection() as conn:
                    cursor = conn.cursor()
                    user_chips_table_name = f"user_chips_{self.table_suffix}"
                    group_state_table_name = f"group_blackjack_state_{self.table_suffix}"

                    # Check for the existence of the user_chips table and its columns
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (user_chips_table_name,))
                    if cursor.fetchone():
                        # Check for the existence of each column before adding
                        cursor.execute("PRAGMA table_info({})".format(user_chips_table_name))
                        columns = [row[1] for row in cursor.fetchall()]
                        if 'nickname' not in columns:
                            cursor.execute(f"ALTER TABLE {user_chips_table_name} ADD COLUMN nickname TEXT")
                        if 'group_jid' not in columns:
                            cursor.execute(f"ALTER TABLE {user_chips_table_name} ADD COLUMN group_jid TEXT")
                        # Add scramble_score column if it doesn't exist
                        if 'scramble_score' not in columns:
                            cursor.execute(f"ALTER TABLE {user_chips_table_name} ADD COLUMN scramble_score INTEGER DEFAULT 0")
                    else:
                        # Create the table with the new column
                        cursor.execute(f"""
                            CREATE TABLE {user_chips_table_name} (
                                jid TEXT PRIMARY KEY,
                                chips INTEGER NOT NULL DEFAULT 1000000,
                                bet_amount INTEGER DEFAULT 0,
                                nickname TEXT,
                                group_jid TEXT,
                                scramble_score INTEGER DEFAULT 0
                            );
                        """)

                    # Check for the existence of the group_blackjack_state table
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (group_state_table_name,))
                    if not cursor.fetchone():
                        cursor.execute(f"""
                            CREATE TABLE {group_state_table_name} (
                                group_id TEXT PRIMARY KEY
                            );
                        """)

                    conn.commit()
            except sqlite3.Error as e:
                logging.error(f"Error setting up database: {e}")

    # Rest of the methods...
    def set_user_nickname(self, jid, nickname):
        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            try:
                with self.create_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"UPDATE {table_name} SET nickname = ? WHERE jid = ?", (nickname, jid))
                    conn.commit()
                    logging.info(f"Nickname set for {jid}: {nickname}")
                
                    # Update user data in the dictionary
                    if jid in self.user_data:
                        self.user_data[jid]['nickname'] = nickname
            except sqlite3.Error as e:
                logging.error(f"Error setting nickname for {jid}: {e}")
    def get_user_nickname_from_db(self, jid):
        table_name = f"user_chips_{self.table_suffix}"
        try:
            with self.create_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT nickname FROM {table_name} WHERE jid = ?", (jid,))
                result = cursor.fetchone()
                return result[0] if result else None
        except sqlite3.Error as e:
            logging.error(f"Error fetching nickname for {jid}: {e}")
            return None
    def update_scramble_score(self, jid, increment=1):
        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            try:
                with self.create_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"UPDATE {table_name} SET scramble_score = scramble_score + ? WHERE jid = ?", (increment, jid))
                    conn.commit()
            except sqlite3.Error as e:
                logging.error(f"Error updating scramble score for {jid}: {e}")

    def get_nickname_scramble_leaderboard(self, group_jid):
        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            try:
                with self.create_connection() as conn:
                    cursor = conn.cursor()
                    # Fetching the nickname and scramble score along with user details
                    cursor.execute(f"SELECT jid, nickname, scramble_score FROM {table_name} WHERE group_jid=?", (group_jid,))
                    leaderboard_data = cursor.fetchall()
                    # Sorting the leaderboard data based on scramble score
                    leaderboard_data.sort(key=lambda x: x[2], reverse=True)  # Sorting by scramble score
                    return leaderboard_data
            except sqlite3.Error as e:
                logging.error(f"Error in get_nickname_scramble_leaderboard: {e}")
        return []
    def get_user_leaderboard(self, group_jid):
        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            try:
                with self.create_connection() as conn:
                    cursor = conn.cursor()
                 # Fetching the scramble score along with other user details
                    cursor.execute(f"SELECT jid, nickname, chips, scramble_score FROM {table_name} WHERE group_jid=?", (group_jid,))
                    leaderboard_data = cursor.fetchall()
                    # Sorting the leaderboard data based on your preference
                    leaderboard_data.sort(key=lambda x: (x[2], x[3]), reverse=True)  # Example: Sorting by chips and then by scramble score
                    return leaderboard_data
            except sqlite3.Error as e:
                logging.error(f"Error in get_user_leaderboard: {e}")
        return []

    def get_user_nickname(self, jid):
        if jid in self.user_data:
            return self.user_data[jid]['nickname']
        return None

    def get_all_group_ids(self):
        with self.lock:
            with self.create_connection() as conn:
                cursor = conn.cursor()
                group_state_table_name = f"group_blackjack_state_{self.table_suffix}"
                cursor.execute(f"SELECT DISTINCT group_id FROM {group_state_table_name}")
                groups = cursor.fetchall()
                return [group[0] for group in groups]

    def reset_all_bets(self, group_jid):
        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            try:
                with self.create_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"UPDATE {table_name} SET bet_amount = 0 WHERE group_jid = ?", (group_jid,))
                    conn.commit()
                    logging.info(f"All bets reset for group {group_jid}.")
            except sqlite3.Error as e:
                logging.error(f"Error resetting bets for group {group_jid}: {e}")

    def add_user_if_not_exists(self, from_jid, group_jid):
        initial_chips = 1000000
        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            try:
                with self.create_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"INSERT OR IGNORE INTO {table_name} (jid, chips, group_jid) VALUES (?, ?, ?)", (from_jid, initial_chips, group_jid))
                    conn.commit()
                    logging.info(f"User {from_jid} added or already exists in the database for group {group_jid}.")
            except sqlite3.Error as e:
                logging.error(f"Error in add_user_if_not_exists for {from_jid}: {e}")

    def get_user_chips(self, jid):
        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            with self.create_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT chips FROM {table_name} WHERE jid = ?", (jid,))
                result = cursor.fetchone()
                return result['chips'] if result else 0

    def update_user_chips(self, jid, chips):
        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            with self.create_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"UPDATE {table_name} SET chips = ? WHERE jid = ?", (chips, jid))
                conn.commit()

    def update_user_chips_and_bet(self, jid, chips, bet_amount):
        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            try:
                with self.create_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"UPDATE {table_name} SET chips = ?, bet_amount = ? WHERE jid = ?", (chips, bet_amount, jid))
                    conn.commit()
            except sqlite3.Error as e:
                logging.error(f"Error updating chips and bet for {jid}: {e}")
    

    def add_chips_to_user(self, jid, chips_to_add):
        if not isinstance(chips_to_add, int):
            try:
                chips_to_add = int(chips_to_add)
            except ValueError:
                raise ValueError("chips_to_add must be an integer")

        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            with self.create_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"UPDATE {table_name} SET chips = chips + ? WHERE jid = ?", (chips_to_add, jid))
                conn.commit()

    def subtract_chips_from_user(self, jid, chips_to_subtract):
        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            with self.create_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT chips FROM {table_name} WHERE jid = ?", (jid,))
                result = cursor.fetchone()
                if result:
                    current_chips = result['chips']
                    new_chips = max(current_chips - chips_to_subtract, 0)
                    cursor.execute(f"UPDATE {table_name} SET chips = ? WHERE jid = ?", (new_chips, jid))
                    conn.commit()

    def set_user_bet(self, jid, bet_amount):
        if bet_amount < 0:
            logging.warning(f"Attempt to set a negative bet amount for {jid}: {bet_amount}")
            return

        table_name = f"user_chips_{self.table_suffix}"
        with self.lock:
            try:
                with self.create_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"UPDATE {table_name} SET bet_amount = ? WHERE jid = ?", (bet_amount, jid))
                    conn.commit()
                    logging.info(f"Bet set for {jid}: {bet_amount}")
            except sqlite3.Error as e:
                logging.error(f"Error setting bet for {jid}: {e}")


    def get_user_bet(self, jid):
        table_name = f"user_chips_{self.table_suffix}"
        try:
            with self.lock:
                with self.create_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT bet_amount FROM {table_name} WHERE jid = ?", (jid,))
                    result = cursor.fetchone()
                    if result:
                        bet_amount = result['bet_amount']
                        print(f"Retrieved bet amount for {jid}: {bet_amount}")
                        return bet_amount
                    else:
                        print(f"No bet found for {jid}")
                        return 0
        except Exception as e:
            print(f"Error retrieving bet for {jid}: {e}")
            return 0



#admin database 
def add_admin(group_id, user_id):
    conn = sqlite3.connect('db.sqlite3')
    curr = conn.cursor()
    curr.execute(f'INSERT INTO admins VALUES (?, ?)', (group_id, user_id))
    conn.commit()
    conn.close()

def remove_admin(group_id, user_id):
    conn = sqlite3.connect('db.sqlite3')
    curr = conn.cursor()
    curr.execute(f'DELETE FROM admins WHERE (group_id = ? AND user_id = ?)', (group_id, user_id))
    conn.commit()
    conn.close()

logging.basicConfig(level=logging.DEBUG)

def is_user_admin(user_id, group_id):
    conn = sqlite3.connect('db.sqlite3')
    curr = conn.cursor()
    curr.execute(f'SELECT * FROM admins WHERE (user_id=? AND group_id = ?)', (user_id, group_id))
    rows = curr.fetchall()

    conn.close()

    if rows == []:
        return False
    else:
        return True

def get_admins(group_id):
    with sqlite3.connect('db.sqlite3') as conn:
        curr = conn.cursor()
        curr.execute('SELECT * FROM admins WHERE group_id=?', (group_id,))
        rows = curr.fetchall()

    if not rows:
        return "No admins found in this group."

    admin_list = "List of Admins:\n"
    for row in rows:
        admin_list += f"- {row[1]}\n"  # Assuming row[1] contains the admin's name or identifier

    return admin_list

###   CUSTOME COMMANDS LIKE WELCOME MESSAGE AND TRIGGER RESPONS DATABASE 
#@101KEK

import sqlite3

class ChatbotDatabase:
    def __init__(self):
        self.connection = sqlite3.connect('bot_data.db', check_same_thread=False)
        self.cursor = self.connection.cursor()
        self.initialize_database()

    def initialize_database(self):
       
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS welcomes (
                                   group_id TEXT PRIMARY KEY,
                                   welcome_message TEXT
                               )''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS word_responses (
                           group_id TEXT,
                           word TEXT,
                           response TEXT
                       )''')
        self.connection.commit()
    
    
    
    def get_custom_command_response(self, group_id, word):
        try:
            self.cursor.execute("SELECT response FROM word_responses WHERE group_id = ? AND word = ?", (group_id, word))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"Error retrieving word response: {e}")
            return None
    def save_word_response(self, group_id, word, response):
        try:
            print(f"Saving word response: Group ID: {group_id}, Word: {word}, Response: {response}")
            print(f"Types - Group ID: {type(group_id)}, Word: {type(word)}, Response: {type(response)}")
            self.cursor.execute("INSERT INTO word_responses (group_id, word, response) VALUES (?, ?, ?)",
                                (group_id, word, response))
            self.connection.commit()
            return "Word response saved successfully."
        except Exception as e:
            return f"Error saving word response: {e}"

    def get_word_responses(self, group_id):
        try:
            self.cursor.execute("SELECT word, response FROM word_responses WHERE group_id = ?", (group_id,))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error retrieving word responses: {e}")
            return []

    def delete_word_response(self, group_id, word):
        try:
            self.cursor.execute("DELETE FROM word_responses WHERE group_id = ? AND word = ?", (group_id, word))
            self.connection.commit()
            if self.cursor.rowcount == 0:
                print(f"No word response found to delete for word '{word}' in group '{group_id}'.")
                return "No such word response found to delete."
            print(f"Deleted word response '{word}' successfully for group '{group_id}'.")
            return "Word response deleted successfully."
        except Exception as e:
            print(f"Error deleting word response: {e}")
            return f"Error deleting word response: {e}"
    def save_welcome(self, group_id, text):
        try:
            text = text.replace('"', "\DQUOTE")
            self.cursor.execute("INSERT INTO welcomes (group_id, welcome_message) VALUES (?, ?)", (group_id, text))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")

    def get_welcome(self, group_id):
        try:
            self.cursor.execute("SELECT welcome_message FROM welcomes WHERE group_id = ?", (group_id,))
            row = self.cursor.fetchone()
            if row:
                return row[0].replace('\DQUOTE', '"')
            else:
                return "No welcome message set."  # Return a default message if no entry is found
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return "An error occurred while retrieving the welcome message."
        # No need for finally block to close connection, as we're using a single persistent connection

    def delete_welcome(self, group_id):
        try:
            self.cursor.execute("DELETE FROM welcomes WHERE group_id = ?", (group_id,))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
    

    

    def get_custom_commands(self, group_id):
        try:
            self.cursor.execute("SELECT trigger, response FROM custom_commands WHERE group_id = ?", (group_id,))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error retrieving custom commands: {e}")
            return []

    def close(self):
        self.connection.close()


#MAKE SURE TO FIX THE CREDIT PATH UNDER YOURPATHHERE
import argparse
from collections import defaultdict
import itertools
import json
import logging
import os
import random
import re
import sqlite3
import string
import sys
import threading
import time
from typing import Union
from bs4 import BeautifulSoup
from termcolor import colored
import openai
import requests
import validators

from blackjack_bot import BlackjackDatabase
from chatbot_db import ChatbotDatabase
from helper_funcs import add_admin, is_user_admin, remove_admin
from kik_unofficial.datatypes.xmpp.roster import FetchRosterResponse, PeersInfoResponse
from kik_unofficial.datatypes.xmpp.xiphias import UsersByAliasResponse, UsersResponse

username = {}

response = {}
import kik_unofficial.datatypes.xmpp.chatting as chatting
from kik_unofficial.client import KikClient
from kik_unofficial.callbacks import KikClientCallback
from kik_unofficial.datatypes.xmpp.errors import SignUpError, LoginError

from kik_unofficial.datatypes.xmpp.sign_up import RegisterResponse
from kik_unofficial.datatypes.xmpp.login import LoginResponse, ConnectionFailedResponse, CaptchaElement, TempBanElement

users = {}
# Define Unicode symbols for card suits

# Specify the path to your SQLite database file (replace 'your_database_file.sqlite3' with your actual path)
db_path = 'blackjack_bot.db'
# Specify a fixed table suffix that remains consistent across bot restarts
table_suffix = 'users'  
DEFAULT_BET_AMOUNT = 0
# Create the Database instance with the provided database file path and table suffix
database = BlackjackDatabase(db_path, table_suffix)
users = {}

card = {'rank': 'A', 'suit': 'Spades'}
def main():
    # The credentials file where you store the bot's login information
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--creds', default='creds.json', help='Path to credentials file')
    args = parser.parse_args()

    # Changes the current working directory to /examples
    if not os.path.isfile(args.creds):
        print("Can't find credentials file.")
        return

    # load the bot's credentials from creds.json
    with open(args.creds, "r") as f:
        creds = json.load(f)

    bot = EchoBot(creds)

def sanitize_filename(filename):
     # Replace or remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', '', filename)
    return sanitized

def randomString(length):
    return ''.join(random.choice(string.ascii_uppercase) for i in range(length)) 
try:
    from urllib.request import urlopen
    from urllib.parse import urlencode
except ImportError:
    from urllib import urlopen, urlencode
# Constants
VIDEO_SIZE_LIMIT = 15728640
VIDEO_COMPRESSION_LIMIT = 5000000
VIDEO_SLEEP_DELAY = 5
from moviepy.editor import VideoFileClip
from pytube import YouTube, Search
import validators
    # create the bot
HEARTS = '♥'
DIAMONDS = '♦'
CLUBS = '♣'
SPADES = '♠'

    
# Configure the logging as usual
logging.basicConfig(
    level=logging.INFO,  # Set your desired log level
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
class EchoBot(KikClientCallback):
    
    DEFAULT_BET_AMOUNT = 0  # Define a default bet amount for the blackjack game
    def __init__(self, creds: dict, database):
        self.bot_display_name = None
        self.pong_list = []
        self.in_blackjack_game = False
        self.game_starter_jid = None  # JID of the user who started the game
        self.deck = []  # The deck for blackjack
        self.player = []  # The player's hand
        self.dealer = []  # The dealer's hand
        self.database = database
        self.game_state = defaultdict(lambda: {'in_game': False, 'deck': [], 'player_hand': [], 'dealer_hand': [], 'current_phase': 'waiting_for_bets', 'bet_amount': 0})
        
       
        # Initialize dictionaries to hold commands and image triggers for each group
        self.db = ChatbotDatabase()
        # Call the reset function during bot initialization
        self.reset_all_games()
        #HEARTBEAT KEEP ALIVE PRIMAL WAY
        self.start_heartbeat()
        username = creds['username']
        password = creds.get('password') or input("Enter your password:")
        # Optional parameters
        device_id = creds['device_id']
        android_id = creds['android_id']

        node = creds.get('node')
        self.client = KikClient(self, username, str(password), node, device_id=device_id, android_id=android_id)
        
        
        self.custom_commands = {}
        # Initialize dictionaries 
        
        #YOUTUBE
        self.search_results = {}  # Dictionary to store search results
        self.awaiting_selection = {}  # Dictionary to track if awaiting selection
        self.game_initiators = {}  # Dictionary to store game initiators
        
       
        self.db_lock = threading.Lock()
        self.user_data = {}
        self.db = BlackjackDatabase(db_path, table_suffix)  # Ensure this is the correct class with the transfer_chips method
        self.database = ChatbotDatabase()
        logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s]: %(message)s")
        self.client.wait_for_messages()
    
        
    

    
    def on_authenticated(self):
        print("Authenticated")
       
       
    

    
    def query_user(self, jid):
        if jid in self.users:
            return self.users[jid]
        else:
            self.client.request_info_of_users(jid)
            while jid not in self.users:
                pass  # You might want to add a timeout or a better waiting mechanism here
            return self.users[jid]



    def get_group_jid_number(jid):
        return jid.split('@')[0][0:-2]

    def send_heartbeat(self, group_jid='ADD JID OR GJID HERE'): #ADD A group_jid OR user_jid
        while True:
            try:
                if group_jid:
                    self.client.send_chat_message(group_jid, " Status Check: Online Ping")#EDIT YOUR MESSAGE
                time.sleep(300)  
            except Exception as e:
                logging.error(f"Heartbeat error: {e}")
    def start_heartbeat(self):
        heartbeat_thread = threading.Thread(target=self.send_heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()    
        
    # This method is called when the bot is fully logged in and setup
    def on_authenticated(self):
        self.client.request_roster() # request list of chat partners
    
  
  #PRIMALS BLACKJACK BOT
    def show_leaderboard(self, group_jid, user_jid):
        leaderboard_data = database.get_user_leaderboard(group_jid)

        if not leaderboard_data:
            return "No data available for this group."

        leaderboard_text = "Group Leaderboard:\n"

        for rank, (jid, nickname, chips, hm_score) in enumerate(leaderboard_data, start=1):
            display_name = nickname if nickname else "'/nickname'"
            leaderboard_text += f"{rank}. {display_name}: {chips} chips\n"

        # Check if user_jid has a nickname set and include it in the leaderboard text
        user_nickname = database.get_user_nickname(user_jid)

        if user_nickname:
            leaderboard_text += f"\nYour nickname: {user_nickname}"

        return leaderboard_text
    def game_help(self, chat_message):
        game_help = (
            "Blackjack Quick Guide:\n\n"
            "LeaderBoard:\n"
            "/nickname <rank name>\n"
            "/leaderboard\n"
            "Game Commands:\n\n /bet <amount> (start game),\n\n /hit (draw card),\n\n /stand (end turn),\n\n "
            "/double (double bet for one card),\n\n /chips (check count),\n\n"
        
            "Goal:\n\n Beat dealer's hand without exceeding 21. Blackjack = 21 points from deal. Tie = bets returned.\n\n"

            "Card Values:\n\n Numbers = face value, Face cards = 10, Aces = 1 or 11.\n\n"

            "Decisions:\n\n 'Hit' if < 11, 'Stand' if ≥ 17.\n 'Double Down' to increase bet with only one more card."
        )
        self.client.send_chat_message(chat_message.group_jid, game_help)
    def initialize_deck(self):
        # Create a standard 52-card deck
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [{'suit': suit, 'rank': rank} for suit in suits for rank in ranks]
        random.shuffle(deck)
        return deck
    def update_user_bet_in_game_state(self, user_jid, group_jid, bet_amount):
        # Initialize the players entry if it doesn't exist
        if group_jid not in self.game_state:
            self.prepare_blackjack_game(group_jid)

        # Initialize the player's state if it doesn't exist
        if user_jid not in self.game_state[group_jid]['players']:
            self.game_state[group_jid]['players'][user_jid] = {
                'hands': [[]],
                'bet_amount': [],
                'active_hand': 0
            }

        # Update the bet amount for the current active hand
        active_hand_index = self.game_state[group_jid]['players'][user_jid]['active_hand']
        if len(self.game_state[group_jid]['players'][user_jid]['bet_amount']) > active_hand_index:
            self.game_state[group_jid]['players'][user_jid]['bet_amount'][active_hand_index] = bet_amount
        else:
            self.game_state[group_jid]['players'][user_jid]['bet_amount'].append(bet_amount)
    
    def process_bet(self, chat_message, bet_amount):
        user_jid = chat_message.from_jid
        group_jid = chat_message.group_jid

        # Check if the game is in the right phase for betting
        if self.game_state[group_jid]['current_phase'] != 'waiting_for_bets':
            self.client.send_chat_message(group_jid, "In the sexy world of blackjack, you can only place your bets at the start of a thrilling new game. ")
            return

        # Update the game state with the user's bet
        self.game_state[group_jid]['players'][user_jid]['bet_amount'] = bet_amount
        self.game_state[group_jid]['current_phase'] = 'waiting_to_start'
        self.client.send_chat_message(group_jid, f"Bet of {bet_amount} accepted from {user_jid}. Type '/bet' <amount>.")
    def check_and_execute_dealer_play(self, group_jid):
        if self.are_all_hands_finished(group_jid):
            self.dealer_play(group_jid)
    def has_user_placed_bet(self, user_jid, group_jid):
        # Check if the group is in the game state
        if group_jid not in self.game_state:
            return False

        # Check if the user is in the players list of the group
        if user_jid not in self.game_state[group_jid].get('players', {}):
            return False

        # Retrieve the bet amount, defaulting to an empty list if not set
        bet_amounts = self.game_state[group_jid]['players'][user_jid].get('bet_amount', [])

        # Check if any bet in the list is greater than 0
        return any(bet > 0 for bet in bet_amounts)
    def reset_all_games(self):
        try:
            all_group_ids = database.get_all_group_ids()
            for group_id in all_group_ids:
                self.game_state[group_id]['in_game'] = False
                self.game_state[group_id]['current_phase'] = 'waiting_for_bets'
                self.game_state[group_id]['player_hand'] = []
                self.game_state[group_id]['dealer_hand'] = []
                self.game_state[group_id]['deck'] = self.initialize_deck()
            print("All games have been reset.")
        except Exception as e:
            print(f"Error resetting games: {e}")
            print("All games have been reset.")

#cards Code
    def display_card(self, card):
        # Map the suit to its corresponding symbol
        suit_symbols = {
            'Hearts': HEARTS,
            'Diamonds': DIAMONDS,
            'Clubs': CLUBS,
            'Spades': SPADES
        }

        # Retrieve the symbol using the suit of the card
        suit_symbol = suit_symbols.get(card['suit'], '')  # Default to an empty string if the suit is not found

        # Return the formatted string representation of the card
        return f"{card['rank']} {suit_symbol}"
    def draw_hand(self, deck):
        return [self.draw_card(deck), self.draw_card(deck)]    
    def draw_card(self, deck):
        if not deck:
            deck.extend(self.initialize_deck())
        return deck.pop()
    def display_hand(self, hand):
        # Use display_card to get the string representation of each card in the hand
        hand_representation = ", ".join(self.display_card(card) for card in hand)
        print(f"Displaying hand: {hand_representation}")  # Debug print
        return hand_representation
    def display_dealer_hand(self, group_jid):
        # Fetch the first card of the dealer's hand from the game state
        dealer_first_card = self.game_state[group_jid]['dealer_hand'][0]

        # Use the display_card method to get the string representation of the first card
        first_card_display = self.display_card(dealer_first_card)

        # The rest of the dealer's hand is not visible to the player (X)
        return f"{first_card_display},X"
    def dealer_play(self, group_jid):
        if not self.are_all_hands_finished(group_jid):
            return  # Return early if any player hand is still active

        dealer_hand = self.game_state[group_jid]['dealer_hand']
        deck = self.game_state[group_jid]['deck']

        while self.calculate_score(dealer_hand) < 17:
            if not deck:
                deck.extend(self.initialize_deck())
                random.shuffle(deck)

            new_card = self.draw_card(deck)
            dealer_hand.append(new_card)

        self.game_state[group_jid]['dealer_hand'] = dealer_hand
        # After dealer play, determine winner for all hands
        for user_jid in self.game_state[group_jid]['players']:
            winner_message = self.determine_winner(group_jid, user_jid)
            self.client.send_chat_message(group_jid, winner_message)
        self.end_blackjack_game(group_jid)  # End the game after dealer's turn

        # Update dealer's hand in game state
        self.game_state[group_jid]['dealer_hand'] = dealer_hand
    def are_all_hands_finished(self, group_jid):
        for player in self.game_state[group_jid]['players'].values():
            if player['active_hand'] < len(player['hands']):
                return False
        return True
    def calculate_score(self, hand):
        values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11}
        score = sum(values[card['rank']] for card in hand)
        num_aces = sum(1 for card in hand if card['rank'] == 'A')
        while score > 21 and num_aces:
            score -= 10
            num_aces -= 1
        return score
    def end_blackjack_game(self, group_jid):
        if group_jid in self.game_state:
            # Reset the game state
            self.game_state[group_jid]['in_game'] = False
            self.game_state[group_jid]['current_phase'] = 'waiting_for_bets'
            self.game_state[group_jid]['deck'] = self.initialize_deck()
            self.game_state[group_jid]['dealer_hand'] = []

            # Reset each player's state in this group
            for user_jid in self.game_state[group_jid]['players']:
                self.game_state[group_jid]['players'][user_jid] = {
                    'hands': [[]],
                    'bet_amount': [],
                    'active_hand': 0
                }
        else:
            logging.warning(f"Group ID {group_jid} not found in game state.")
    def prepare_blackjack_game(self, group_jid, user_jid):
        # Initial setup for the game state
        self.game_state[group_jid] = {
            'in_game': False,
            'game_starter_jid': user_jid,
            'deck': self.initialize_deck(),
            'player_hand': [],  # This might be deprecated if using 'hands' in 'players'
            'dealer_hand': [],
            'current_phase': 'waiting_for_bet',
            'players': {}  # Initialize an empty dictionary for players
        }

        # Setup for each player
        self.game_state[group_jid]['players'][user_jid] = {
            'hands': [[]],  # A list of hands, each hand is a list of cards
            'bet_amount': [],  # Initially empty, will be filled when the player places a bet
            'active_hand': 0,  # Index of the currently active hand
            'group_jid': group_jid  # Set the group_jid for the user
        }
    def start_blackjack_game(self, chat_message):
        group_jid = chat_message.group_jid
        user_jid = chat_message.from_jid

        # Ensure the game state for this group is initialized
        if group_jid not in self.game_state:
            self.prepare_blackjack_game(group_jid, user_jid)

        # Check if a game is already in progress
        if self.game_state[group_jid]['in_game']:
            self.client.send_chat_message(group_jid, "A game is already in progress. Finish it before starting a new one.")
            return

        # Check the user's bet for this specific group
        bet_amounts = self.game_state[group_jid]['players'][user_jid].get('bet_amount', [])
        user_chips = database.get_user_chips(user_jid)

        # Check if at least one bet is placed and the user has enough chips
        if bet_amounts and all(bet <= user_chips for bet in bet_amounts):
            print(group_jid,(f"User Chips Before Deduction: {user_chips}"))
            for bet in bet_amounts:
                database.subtract_chips_from_user(user_jid, bet)
                user_chips_after_deduction = database.get_user_chips(user_jid)
                print(group_jid,(f"User Chips After Deduction: {user_chips_after_deduction}"))
            self.start_new_blackjack_round(group_jid, user_jid)
        else:
            self.client.send_chat_message(group_jid, "Please place a bet using '/bet <amount>'")
    
    def start_new_blackjack_round(self, group_jid, user_jid):
        # Set the game as in progress for this group
        self.game_state[group_jid]['in_game'] = True
        self.game_state[group_jid]['deck'] = self.initialize_deck()
        self.game_state[group_jid]['players'][user_jid]['hands'] = [self.draw_hand(self.game_state[group_jid]['deck'])]
        self.game_state[group_jid]['dealer_hand'] = self.draw_hand(self.game_state[group_jid]['deck'])
        self.game_state[group_jid]['current_phase'] = 'player_turn'

        # Send game details to the group chat
        game_message = "BlackJack Game Started\n"
        game_message += f"Dealer's Hand: {self.display_card(self.game_state[group_jid]['dealer_hand'][0])},X\n"
        game_message += f"Your Hand: {self.display_hand(self.game_state[group_jid]['players'][user_jid]['hands'][0])}\n"
        game_message += "Type '/hit' or '/stand' or '/double'."
        self.client.send_chat_message(group_jid, game_message)
    def stand(self, chat_message):
        group_jid = chat_message.group_jid
        if not self.game_state[group_jid]['in_game']:
            self.client.send_chat_message(group_jid, "No active blackjack game. Start one with /bj.")
            return

        self.dealer_play(group_jid)
        winner_message = self.determine_winner(group_jid, chat_message.from_jid)
        self.client.send_chat_message(group_jid, winner_message)
        self.end_blackjack_game(group_jid)  # Reset the game state
    def hit(self, chat_message):
        group_jid = chat_message.group_jid
        user_jid = chat_message.from_jid

        # Check if a blackjack game is active
        if not self.game_state[group_jid]['in_game']:
            self.client.send_chat_message(group_jid, "No active blackjack game. Start one with /bet <amount>.")
            return

        # Access the game deck and the player's current hand
        deck = self.game_state[group_jid]['deck']
        player_hand = self.game_state[group_jid]['players'][user_jid]['hands'][0]

        # Draw a new card and append it to the active hand
        new_card = self.draw_card(deck)
        player_hand.append(new_card)

        # Display the dealer's first card and a placeholder for the second card
        dealer_first_card = self.game_state[group_jid]['dealer_hand'][0]
        dealer_hand_display = f"Dealer's Hand: {self.display_card(dealer_first_card)}, Face Down"

        # Display the updated player hand
        player_hand_display = f"Your Hand: {self.display_hand(player_hand)}"

        # Check if player busts
        if self.calculate_score(player_hand) > 21:
            game_message = f"{dealer_hand_display}\n{player_hand_display}\nBusted!"
            self.client.send_chat_message(group_jid, game_message)

            # Determine the winner (or in this case, the loss)
            winner_message = self.determine_winner(group_jid, user_jid)
            self.client.send_chat_message(group_jid, winner_message)

            # End and reset the game
            self.end_blackjack_game(group_jid)
        else:
            game_message = f"{dealer_hand_display}\n{player_hand_display}\nOptions: /hit, /stand"
            self.client.send_chat_message(group_jid, game_message)
    def double_down(self, chat_message):
        user_jid = chat_message.from_jid
        group_jid = chat_message.group_jid

        if self.game_state[group_jid]['in_game']:
            player = self.game_state[group_jid]['players'][user_jid]
            active_hand = player['hands'][player['active_hand']]

            # Validate if the player can double down (only with two cards in hand)
            if len(active_hand) == 2:
                bet_amount = player['bet_amount'][player['active_hand']]

                if database.get_user_chips(user_jid) >= bet_amount:
                    # Double the bet and update the bet amount in the game state
                    doubled_bet_amount = 2 * bet_amount
                    player['bet_amount'][player['active_hand']] = doubled_bet_amount

                    # Deduct the additional bet amount from the user's chips
                    database.subtract_chips_from_user(user_jid, bet_amount)

                    new_card = self.draw_card(self.game_state[group_jid]['deck'])
                    active_hand.append(new_card)

                    # Check for bust after drawing the new card
                    if self.calculate_score(active_hand) > 21:
                        # Player is busted, end the game
                        self.client.send_chat_message(group_jid, f"Doubled down. Busted with hand: {self.display_hand(active_hand)}")
                        winner_message = self.determine_winner(group_jid, user_jid)
                        self.client.send_chat_message(group_jid, winner_message)
                        self.end_blackjack_game(group_jid)
                    else:
                        # Continue the game
                        self.client.send_chat_message(group_jid, f"Doubled down. Your new hand: {self.display_hand(active_hand)}")
                    
                        # If it's the last hand, proceed to dealer play
                        if player['active_hand'] + 1 >= len(player['hands']):
                            self.dealer_play(group_jid)
                            winner_message = self.determine_winner(group_jid, user_jid)
                            self.client.send_chat_message(group_jid, winner_message)
                            self.end_blackjack_game(group_jid)
                else:
                    self.client.send_chat_message(group_jid, "You do not have enough chips to double down.")
            else:
                self.client.send_chat_message(group_jid, "Can only double down at the start of your turn with two cards.")

    def place_bet(self, chat_message, bet_amount):
        user_jid = chat_message.from_jid
        group_jid = chat_message.group_jid

        # Check if the user has enough chips to place the bet
        user_chips = database.get_user_chips(user_jid)
        if user_chips is None:
            user_chips = self.DEFAULT_BET_AMOUNT  # Assign a default chip count if the user has no record yet

        if bet_amount > 0 and user_chips >= bet_amount:
            # Deduct the bet amount from the user's chips
            database.subtract_chips_from_user(user_jid, bet_amount)

            # Update the game state with the bet amount
            self.update_user_bet_in_game_state(user_jid, group_jid, bet_amount)

            # Inform the user that the bet has been placed
            self.client.send_chat_message(group_jid, f"Bet of {bet_amount} chips placed. Your remaining chips: {user_chips - bet_amount}")
        else:
            # Inform the user they don't have enough chips to place the bet
            self.client.send_chat_message(group_jid, "Invalid bet amount or insufficient chips.")
    
    
    def determine_winner(self, group_jid, user_jid):
        player_hands = self.game_state[group_jid]['players'][user_jid]['hands']
        dealer_hand = self.game_state[group_jid]['dealer_hand']
        dealer_score = self.calculate_score(dealer_hand)
        dealer_hand_display = self.display_hand(dealer_hand)
        overall_outcome = ""

        for i, player_hand in enumerate(player_hands):
            bet_amount = self.game_state[group_jid]['players'][user_jid]['bet_amount'][i]
            player_score = self.calculate_score(player_hand)
            player_hand_display = self.display_hand(player_hand)

            if player_score > 21:
                # Player busted, subtract bet amount
                outcome = f"Busted!\nYour Hand: {player_hand_display}\nDealer's Hand: {dealer_hand_display}\nAmount Lost: {bet_amount}"
            elif dealer_score > 21 or player_score > dealer_score:
                # Player wins, add winnings
                winnings = bet_amount * 2  # Corrected winnings calculation
                database.add_chips_to_user(user_jid, winnings + bet_amount)  # Add both original bet and winnings
                outcome = f"You win!\nYour Hand: {player_hand_display}\nDealer's Hand: {dealer_hand_display}\nWinnings: {winnings}"
            elif dealer_score > player_score:
                # Dealer wins, subtract bet amount
                outcome = f"Dealer wins.\nDealer's Hand: {dealer_hand_display}\nAmount Lost: {bet_amount}"
            else:
                # It's a tie, return the bet amount
                database.add_chips_to_user(user_jid, bet_amount)  # Add the original bet back
                outcome = f"It's a tie!\nYour Hand: {player_hand_display}\nDealer's Hand: {dealer_hand_display}"

            overall_outcome += outcome + "\n"

        # Get and display the new chip count
        new_chip_count = database.get_user_chips(user_jid)
        overall_outcome += f"Your new chip count: {new_chip_count}"

        return overall_outcome
    #Yooutube downloader sort of 
    def download_and_send_youtube_video(self, input_text, group_jid):
        try:
            # Check if input text is a valid URL
            if validators.url(input_text):
                yt = YouTube(input_text)
            else:
                # If not a URL, perform a search and list the top results
                search = Search(input_text)
                if not search.results:
                    return "No search results found for the query."

                # Store the search results in the 'self.search_results' dictionary
                self.search_results[group_jid] = search.results[:5]

                # Create a list of search results as text
                search_results_text = []
                for i, video in enumerate(self.search_results[group_jid]):
                    title = video.title
                    search_results_text.append(f"{i+1}. {title}")

                # Construct the response message with the search results
                response = "Search results:\n" + '\n'.join(search_results_text) + "\nReply with the number to download."
                return response

            video_stream = yt.streams.filter(file_extension="mp4").first()
            if video_stream:
                download_path = r'FILEPATHHERE'
                if not os.path.exists(download_path):
                    os.makedirs(download_path)

                # Sanitize the filename
                safe_filename = sanitize_filename(yt.title)
                video_path = os.path.join(download_path, safe_filename + ".mp4")
                video_stream.download(output_path=download_path, filename=safe_filename + ".mp4")

                # Send the video file
                self.client.send_video_message(group_jid, video_path)

                # Wait 1 second before deleting the file
                time.sleep(1)

                # Delete the video file after sending
                if os.path.exists(video_path):
                    os.remove(video_path)

                # Return information about the video
                video_info = f"{yt.title}\nLength: {int(yt.length / 60)} minutes\nViews: {yt.views}"
                return video_info
            else:
                return "No suitable video stream found."
        except Exception as e:
            # In case of an exception, check if the file exists and delete it
            if os.path.exists(video_path):
                os.remove(video_path)
            return f"An error occurred: {e}"
    def search_youtube_and_list_options(self, query, group_jid):
        try:
            search = Search(query)
            if not search.results:
                return "No search results found for the query."

            # Store top 5 results in the 'self.search_results' dictionary
            self.search_results[group_jid] = search.results[:5]

            # Create a list of search results as text
            search_results_text = []
            for i, video in enumerate(self.search_results[group_jid]):
                title = video.title
                search_results_text.append(f"{i+1}. {title}")

            # Construct the response message with the search results
            response = "Search results:\n" + '\n'.join(search_results_text) + "\nReply with the number to download."
            return response
        except Exception as e:
            return f"An error occurred: {e}"
    def on_video_received(self, response: chatting.IncomingVideoMessage):
        if not response.group_jid:
            print(f"PM Video message was received from {response.video_url}")
        else:
            print(f"Group Video message was received from {response.group_jid}")
    def compress_or_trim_video(self, video_path):
        clip = VideoFileClip(video_path)

        # Option 1: Compress the video
        # Adjust target size (in bytes) as needed
        target_size = 14000000  # For example, 14MB
        compression_ratio = target_size / os.path.getsize(video_path)
        clip_resized = clip.resize(width=int(clip.w * compression_ratio ** 0.5))

        # Option 2: Trim the video
        # Adjust max_duration (in seconds) as needed
        max_duration = 60  # For example, 60 seconds
        if clip.duration > max_duration:
            clip_resized = clip.subclip(0, max_duration)

        # Save the compressed or trimmed video
        new_video_path = os.path.splitext(video_path)[0] + "_compressed.mp4"
        clip_resized.write_videofile(new_video_path)

        clip.close()
        return new_video_path
    
    def handle_commands(self, command, command_parts, chat_message):
        if command == "/save":
            try:
                _, word, response = chat_message.body.split(' ', 2)
                result = self.database.save_word_response(chat_message.group_jid, word, response)
                self.client.send_chat_message(chat_message.group_jid, result)
            except ValueError:
                self.client.send_chat_message(chat_message.group_jid, "Invalid format. Use /save [word] [response]")

        elif command == "/listsave":
            responses = self.database.get_word_responses(chat_message.group_jid)
            response_text = "\n".join([f"{word}/{response}" for word, response in responses])
            self.client.send_chat_message(chat_message.group_jid, response_text or "No saved word responses.")

        elif command == "/deletesave":
            if len(command_parts) > 1:
                word_to_delete = command_parts[1]
                result = self.database.delete_word_response(chat_message.group_jid, word_to_delete)
                self.client.send_chat_message(chat_message.group_jid, result)
            else:
                self.client.send_chat_message(chat_message.group_jid, "Please specify a word to delete. Format: /deletesave [word]")
    
    # This method is called when the bot receives a chat message in a group
    def on_group_message_received(self, chat_message: chatting.IncomingGroupChatMessage):
        separator = colored("--------------------------------------------------------", "cyan")
        group_message_header = colored("[+ GROUP MESSAGE +]", "cyan")
        game_log_header = colored("[+ BLACKJACK GAME LOG +]", "green")

        print(separator)
        print(group_message_header)
        print(colored(f"From AJID: {chat_message.from_jid}" "yellow"))
        print(colored(f"From group: {chat_message.group_jid}", "yellow"))
        print(colored(f"Says: {chat_message.body}", "yellow"))

        # Check if the message is related to the Blackjack game
        if chat_message.group_jid in self.game_state:
            print(separator)
            print(game_log_header)
            print(colored(f"Game State: {self.game_state[chat_message.group_jid]}", "green"))

        print(separator)

        body = chat_message.body.split()
        command = body[0].lower() if body else ""
        group_jid = chat_message.group_jid
        user_jid = chat_message.from_jid
        message_body = chat_message.body
        command_parts = chat_message.body.strip().split()
        command = command_parts[0].lower() if command_parts else ""

        
        
        if command in ["/save", "/listsave", "/deletesave"]:
            self.handle_commands(command, command_parts, chat_message)
            return  # Stop further processing after handling a command
        if command == "/save":
            try:
                _, word, response = chat_message.body.split(' ', 2)
                result = self.database.save_word_response(chat_message.group_jid, word, response)
                self.client.send_chat_message(chat_message.group_jid, result)
            except ValueError:
                self.client.send_chat_message(chat_message.group_jid, "Invalid format. Use /save [word] [response]")
       
        # If the message isn't a command, check for word triggers
        for word in chat_message.body.strip().lower().split():
            word_response = self.database.get_custom_command_response(chat_message.group_jid, word)
            if word_response:
                self.client.send_chat_message(chat_message.group_jid, word_response)
                break  # Stop after finding the first matching word
        
        
        
        # Assuming you want to display the leaderboard when requested
        if command == "/leaderboard":
            leaderboard_message = self.show_leaderboard(group_jid, user_jid)
            self.client.send_chat_message(group_jid, leaderboard_message)
        
        if chat_message.body.lower() == "help":
            with open("help.txt","r") as f:
                self.client.send_chat_message(chat_message.group_jid, f.read())
            return
        
            
        
            
        database.add_user_if_not_exists(user_jid, group_jid)

        if group_jid not in self.game_state:
            self.prepare_blackjack_game(group_jid, user_jid)

        in_game = self.game_state[group_jid]['in_game'] if group_jid in self.game_state else False

        if message_body.startswith("/bet "):
            parts = message_body.split()
            if len(parts) == 2 and parts[1].isdigit():
                bet_amount = int(parts[1])
                self.update_user_bet_in_game_state(user_jid, chat_message.group_jid, bet_amount)
                self.client.send_chat_message(group_jid, f"Bet of {bet_amount} chips placed.")
        
                # Start the game with "/sbj" after placing the bet
                if self.has_user_placed_bet(user_jid, group_jid):
                    self.start_blackjack_game(chat_message)
        
        if "/eg" in message_body:
            self.end_blackjack_game(group_jid)

        elif "/chips" in chat_message.body:
            # Ensure the user exists in the database
            database.add_user_if_not_exists(user_jid, group_jid)

            # Ensure the user has a minimum number of chips
            user_chips = database.get_user_chips(user_jid)
            if user_chips == 0:
                database.update_user_chips(user_jid, 1000)  # Set chips to 1000 if current count is 0
                user_chips = 1000  # Update the local variable to reflect the new chip count

            # Send the updated chip count to the user
            self.client.send_chat_message(group_jid, f"Your chip count: {user_chips} chips")
        elif "/nickname" in chat_message.body:
            # Command to set user's nickname
            try:
                _, nickname = chat_message.body.split(maxsplit=1)
                if len(nickname) > 14:
                    self.client.send_chat_message(group_jid, "Hey there,  keep the nickname under 14 characters.")
                elif not re.match("^[A-Za-z0-9]+$", nickname):
                    self.client.send_chat_message(group_jid, "Hey there, hottie! try again ")
                else:
                    database.set_user_nickname(user_jid, nickname)
                    self.client.send_chat_message(group_jid, f"Mmm, well hello there, {nickname}! ")
            except ValueError:
                self.client.send_chat_message(group_jid, "You can set a  nickname using the command /setnickname followed by your preferred nickname. ")
        if "/gamehelp" in message_body:
            self.game_help(chat_message)
        elif in_game and self.game_state[group_jid]['current_phase'] == 'player_turn':
            if "/hit" in message_body:
                self.hit(chat_message)
            elif "/stand" in message_body:
                self.stand(chat_message)
            elif "/double" in message_body:
                self.double_down(chat_message)
            
        
        
        elif chat_message.body.lower().startswith("/profile"):
            try:
                _, requested_username = chat_message.body.split(maxsplit=1)
                requested_user_jid = self.client.get_jid(requested_username)  # Convert username to JID
                # Rest of your code to fetch the profile using requested_user_jid
            except ValueError:
                self.client.send_chat_message(group_jid, "Usage: /profile <username>")
        message = str(chat_message.body.lower())        
        if is_user_admin(chat_message.from_jid, chat_message.group_jid):
            is_admin = True
            is_superadmin = False
        else:
            is_admin = False
            is_superadmin = False
        
        if message.startswith("/listsave"):
            is_admin
            responses = self.database.get_word_responses(chat_message.group_jid)
            response_text = "\n".join([f"{word}/{response}" for word, response in responses])
            self.client.send_chat_message(chat_message.group_jid, response_text or "No saved word responses.")
        if message.startswith("/deletesave"):
            is_admin
            if len(command_parts) > 1:
                word_to_delete = command_parts[1]
                result = self.database.delete_word_response(chat_message.group_jid, word_to_delete)
                self.client.send_chat_message(chat_message.group_jid, result)
            else:
                self.client.send_chat_message(chat_message.group_jid, "Please specify a word to delete. Format: /deletesave [word]")
        
        
        
        if message.startswith("/getw"):
            is_admin
            welcome_message = self.database.get_welcome(chat_message.group_jid)
            if welcome_message:
                self.client.send_chat_message(chat_message.group_jid, f"Welcome message is: {welcome_message}")
            else:
                self.client.send_chat_message(chat_message.group_jid, "No welcome message set.")
        if message.startswith("/deletew"):
            is_admin
            self.database.delete_welcome(chat_message.group_jid)
            self.client.send_chat_message(chat_message.group_jid, "Welcome message deleted.")
        if message.startswith("/welcome"):
            is_admin
            welcome_message = ' '.join(command_parts[1:])
            self.database.save_welcome(chat_message.group_jid, welcome_message)
            self.client.send_chat_message(chat_message.group_jid, "Welcome message set.")
    def on_group_status_received(self, response: chatting.IncomingGroupStatus):
        print(self.client.request_info_of_users(response.status_jid))
        if re.search(" has promoted ", str(response.status)):
            add_admin(response.group_jid, response.status_jid)

        elif re.search(" has removed admin status from ", str(response.status)):
            remove_admin(response.group_jid, response.status_jid)

        elif re.search(" from this group$", str(response.status)) or re.search("^You have removed ", str(response.status)) or re.search(" has banned ", str(response.status)):
            try:
                remove_admin(response.group_jid, response.status_jid)
            except:
                pass

        elif re.search(" has left the chat$", str(response.status)):
            try:
                remove_admin(response.group_jid, response.status_jid)
            except:
                pass

        elif re.search(" has joined the chat$", str(response.status)) or re.search(" has added you to the chat$", str(response.status)):
                welcome_message = self.database.get_welcome(response.group_jid,)
                if welcome_message:
                    self.client.send_chat_message(response.group_jid, welcome_message)
    
    # This method is called if a captcha is required to login
    def on_login_error(self, login_error: LoginError):
        if login_error.is_captcha():
            login_error.solve_captcha_wizard(self.client)


if __name__ == '__main__':
    main()
    
  
   
# Now you can use the logger to log messages with your custom format
    
    creds_file = "creds.json"

    # Check if the credentials file is in the current working directory, otherwise change directory
    if not os.path.isfile(creds_file):
        os.chdir("FILEPATHHERE")

    # Load the bot's credentials from creds.json
    with open(creds_file) as f:
        creds = json.load(f)
    callback = EchoBot(creds, database)
    
    


ENJOY #.0000 FOR BOT ROOM 
