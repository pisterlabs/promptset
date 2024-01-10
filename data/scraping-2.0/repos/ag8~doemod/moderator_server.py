import asyncio
import openai
import qrcode
import random
import re
import requests
import socket
import string
import subprocess
import sys
import websockets

from http.server import SimpleHTTPRequestHandler, HTTPServer
from typing import Dict, List
from time import sleep

from speech_recognizer import SpeechRecognizer

client = openai.OpenAI()
try:
    client.models.list()
except openai.AuthenticationError:
    print(
        "OpenAI API key not found. Please create a key at https://platform.openai.com/account/api-keys, then set the OPENAI_API_KEY environment variable (https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key)"
    )
    sys.exit(1)
except openai.APIConnectionError:
    print("Couldn't connect to the OpenAI API. Check your internet connection?")
    sys.exit(1)

def say(text, interruptible=False):
    if interruptible:
        return subprocess.Popen(["say", text])
    else:
        subprocess.run(["say", text])


class ModeratorServer:
    def __init__(self):
        self.game_running: bool = False
        self.speech_recognizer: SpeechRecognizer = SpeechRecognizer()
        self.player_names: [str] = []
        self.question_number: int = 0
        self.current_question = {}
        self.is_tossup: bool = True
        self.saying_process: subprocess.Popen = None
        self.current_buzzer: str = None
        self.bonus_player: str = None
        self.scores: Dict[str, int] = {}  # Maps player name to score
        self.buzzed_this_question: List[str] = []  # List of players who have buzzed on this question

    @staticmethod
    def get_question():
        url = "https://scibowldb.com/api/questions/random"

        # Make a GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            return response.json().get('question')
        else:
            # Handle errors (e.g., network issues, server errors)
            return f"Error: Unable to fetch question (reason: {response.reason})"
    
    def next_question(self):
        self.question_number += 1
        if self.question_number > 20:
            self.end_round()
        print(f"Question {self.question_number}")
        self.is_tossup = True
        self.bonus_player = None
        self.buzzed_this_question = []
        
        self.current_question = self.get_question()
        self.read_question()

    def read_question(self):
        question_number = self.question_number
        current_question = self.current_question

        if self.is_tossup:
            say(f"Tossup number {question_number} is in {current_question['category']}; {current_question['tossup_format']}.")
        else:
            say(f"Your bonus is in {current_question['category']}; {current_question['bonus_format']}.")

        self.clear_buzzer()
        type = "tossup" if self.is_tossup else "bonus"
        question_text = current_question[f'{type}_question']
        question_text = re.sub("\[(?:[a-zA-Z]+-)*[A-Z]+(?:-[a-zA-Z]+)*\]", "", question_text)  # remove bracketed phonetic spellings
        question_text = re.sub("\(read as: (?:[a-zA-Z]+-)*[A-Z]+(?:-[a-zA-Z]+)*\)", "", question_text)  # remove "read as" phonetic spellings
        question_text = re.sub("`[^`]+` ?\(read as: ([^\)]+)\)", r"\1", question_text, flags=re.IGNORECASE)  # "read as" equations

        if ": 1)" in question_text:  # list of items; add pause between each numbered item
            question_text = re.sub("[:,;] ([0-9]+)\)", r". [[slnc 500]] \1 [[slnc 250]]", question_text)

        if current_question[f'{type}_format'] == "Multiple Choice":
            # Add pauses between the choices
            question_text = re.sub("\n\(?([W-Z])\)", r". [[slnc 500]] \1 [[slnc 250]]", question_text)

        self.saying_process = say(question_text, interruptible=True)
        return_code = self.saying_process.wait()
        if return_code == 0:
            if self.is_tossup:
                self.countdown(5)
                if self.current_buzzer is None and self.current_question == current_question and self.is_tossup:
                    self.current_buzzer = "locked"

                    say(f"That's time. The correct answer was {self.current_question['tossup_answer']}.")
                    print(end='     \r')
                    self.next_question()
            else:
                self.countdown(20)
                if self.current_buzzer is None and self.current_question == current_question:
                    self.current_buzzer = "locked"
                    say(f"That's time. The correct answer was {self.current_question['bonus_answer']}.")
                    print(end='     \r')
                    self.next_question()
    
    def valid_buzz(self, buzzer):
        if self.game_running is False:
            return False
        elif self.current_buzzer is not None:  # buzzer locked out
            return False
        elif self.is_tossup:
            return buzzer not in self.buzzed_this_question  # can't buzz twice
        else:
            return buzzer == self.bonus_player
    
    def handle_buzz(self, buzzer):
        interrupt = False

        if self.saying_process.poll() is None:  # Interrupt
            interrupt = True
            self.saying_process.kill()
            self.saying_process = None
            if self.is_tossup:
                say("Interrupt!")
        
        self.buzzed_this_question.append(buzzer)
        say(buzzer + ".")

        type = "tossup" if self.is_tossup else "bonus"

        voice_input = self.speech_recognizer.record_and_transcribe(
            f"{self.current_question.get(f'{type}_question')}\nANSWER: {self.current_question.get(f'{type}_answer')}.\nRESPONSE:"
        )

        # Get the correct answer
        correct_answer = self.current_question.get(f'{type}_answer')

        if self.check_answer(voice_input, correct_answer,
                                self.current_question.get(f'{type}_question'),
                                self.current_question.get(f'{type}_format')):
            say(f"{correct_answer} is correct!")
            if self.is_tossup:
                self.is_tossup = False
                self.bonus_player = buzzer
                self.scores[buzzer] += 4
                self.read_question()
            else:
                self.scores[buzzer] += 10
                self.next_question()
        else:
            if interrupt and self.is_tossup:  # interrupt; lose points
                self.scores[buzzer] -= 4
            
            if len(self.buzzed_this_question) < 2 and self.is_tossup:
                say("Incorrect.")
                if interrupt:
                    say(f"I'll re-read for the other player{'s' * (len(self.player_names) > 2)}")
                    self.read_question()
                else:
                    self.clear_buzzer()
                    self.countdown(5)
                    if self.current_buzzer is None and self.is_tossup:
                        self.current_buzzer = "locked"
                        say(f"That's time. The correct answer was {correct_answer}.")
                        print(end='     \r')
                        self.next_question()
            else:
                say(f"Incorrect; the correct answer was {correct_answer}. Moving on.")
                self.next_question()

    def clear_buzzer(self):
        self.current_buzzer = None
    
    def countdown(self, seconds):
        for i in range(seconds, -1, -1):
            if self.current_buzzer is None:
                if i == 0:
                    print("Time!", end=' \r')
                else:
                    print(i, end=' \r')
                    sleep(1)
            else:
                print(end='     \r')
                return

    @staticmethod
    def check_answer(voice_input, correct_answer, question_text, question_type):
        def clean_answer(answer):
            return answer.translate(str.maketrans('', '', string.punctuation)).strip().lower()

        given_answer = clean_answer(voice_input)
        if given_answer == "":
            say("Stall!")
            return False

        if given_answer == "why":
            given_answer = "y"
        elif given_answer in ["ze", "see"]:
            given_answer = "z"
        
        # Remove solution from answer
        solution = re.search("\(Solution: [^)]+\)", correct_answer)
        if solution:
            solution = solution[0]
            correct_answer = correct_answer.replace(solution, "")
        
        # Check for acceptable answer
        if len(re.findall("\(ACCEPT:[^)]+\)", correct_answer)) == 1:
            acceptable_answer = re.search("\(ACCEPT:[^)]+\)", correct_answer)[0][8:-1]
            if given_answer == clean_answer(acceptable_answer):
                return True
        correct_answer = re.sub("\(ACCEPT:[^)]+\)", "", correct_answer)

        # delete everything after a newline
        correct_answer = correct_answer.split("\n")[0]

        if question_type == "Multiple Choice":
            if given_answer in ['w', 'x', 'y', 'z']:  # if it's just a letter
                return given_answer == correct_answer[0].lower()
            elif given_answer == clean_answer(correct_answer[1:]):  # matches everything after the letter
                return True
        
        if given_answer == clean_answer(correct_answer):
            return True


        print(f"You said, {voice_input}. Processed answer is {given_answer}.")
        print("Calling GPT...")

        if question_type == "Multiple Choice":
            prompt = f"You are evaluating an answer for Science Bowl. The question was: ```\n{question_text}\n```. The correct answer is `{correct_answer}`. According to voice transcription software, the player said `{voice_input}`. Should this answer be counted? Saying just the letter of the correct choice is considered correct and should be counted. If the player gave an answer in words, then the player must have given the correct answer, word for word, for the answer to be counted. (The transcription is phonetic, so also count homophones of the correct answer.) Respond only YES or NO. Say YES if the answer should be accepted, and NO if the answer should not be accepted."
        else:
            prompt = f"You are evaluating an answer for Science Bowl. The question was: ```\n{question_text}\n```. The correct answer is `{correct_answer}`. According to voice transcription software, the player said `{voice_input}`. Should this answer be counted? Is it essentially the correct answer, or scientifically also correct? (The transcription is phonetic, so also count homophones of a correct answer.) Respond only YES or NO. Say YES if the answer should be accepted, and NO if the answer should not be accepted. Say only YES or NO, and nothing else."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1,  # YES or NO are each a single token
            temperature=0,
            logit_bias={14331: 50, 9173: 50},  # suppress everything except YES and NO
        )

        result = response.choices[0].message.content

        print(result)

        if result == "YES":
            return True
        elif result == "NO":
            if solution:
                print(solution)
            return False
        else:
            print("Not sure if this answer is correct")
            print(f"Result is {result}")
            return random.random() < 0.3
        
    def print_scores(self):
        print("Scores:")
        for name, score in self.scores.items():
            print(f"{name}: {score}")
        print()

    def run(self):
        print("Welcome to \033[1mdoemod\033[0m!")

        loop = asyncio.get_event_loop()

        # WebSocket handling logic
        async def buzz_handler(websocket, _):
            if self.game_running:
                return
            name = await websocket.recv()
            if name not in self.player_names:
                self.player_names.append(name)
            print(f"{name} from {websocket.remote_address[0]} has joined the game.")
            while True:
                try:
                    message = await websocket.recv()
                    if message == "buzz":
                        if self.valid_buzz(name):
                            self.current_buzzer = name
                            loop.run_in_executor(None, self.handle_buzz, name)
                    else:
                        print("unknown message:", message)
                except websockets.exceptions.ConnectionClosedError:
                    print(f"{name} from {websocket.remote_address[0]} has disconnected.")
                    break

        # Start WebSocket server
        start_websocket_server = websockets.serve(buzz_handler, "", 12348)
        loop.run_until_complete(asyncio.gather(start_websocket_server, return_exceptions=True))

        directory = "srv"
        class DirectoryHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)
            def log_message(self, format, *args):
                pass

        # Start HTTP server
        http_server = HTTPServer(("", 80), DirectoryHandler)
        loop.run_in_executor(None, http_server.serve_forever)

        print(f"Serving at http://{socket.gethostbyname_ex(socket.gethostname())[-1][-1]}.")
        qr = qrcode.QRCode()
        qr.add_data(f"http://{socket.gethostbyname_ex(socket.gethostname())[-1][-1]}/")
        qr.print_ascii()

        def wait_for_start():
            print("Press enter to begin round")
            input("Waiting for players...\n")
            print("Starting...")
            self.start_round()

        loop.run_in_executor(None, wait_for_start)

        try:
            loop.run_forever()
        finally:
            http_server.shutdown()
            loop.close()

    
    def start_round(self):
        # Initialize all scores to zero
        self.scores = {player_name: 0 for player_name in self.player_names}
        self.game_running = True
        self.next_question()

    def end_round(self):
        # Construct the announcement
        announcement = "And that's the round! [[slnc 500]] "
        # Sort the dictionary by scores in increasing order
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1])
        for name, score in sorted_scores[:-1]:  # Go through all but the highest score
            announcement += f"{name} with {score} points. [[slnc 500]] "

        # Add a longer pause before announcing the first place
        announcement += "[[slnc 1000]] And in first place, "
        announcement += f"{sorted_scores[-1][0]} with {sorted_scores[-1][1]} points! Congratulations to the winner!"

        say(announcement)
        self.print_scores()

        sys.exit(0)

if __name__ == "__main__":
    server = ModeratorServer()
    server.run()
