from typing import List, Tuple
import time
import requests
from selenium.webdriver.common.by import By
from src.connections.prompts import prompt_factory
from src.connections.selectors import Selector as S
from src.general.controller import BrowserController
from src.general.config import OPENAI_API_KEY
from src.general.error import ApiRequestError
from src.general.states import GameState
from src.general.util import log_args_retval


class ConnectionsController(BrowserController):
    def __init__(self) -> None:
        super().__init__("connections")
        self.driver.get("https://www.nytimes.com/games/connections")
        self.driver.find_element(By.CLASS_NAME, S.START_BUTTON.value).click()
        time.sleep(2)  # sleep as next page loading
        self.driver.find_element(By.ID, S.CLOSE_BUTTON.value).click()
        self.word_to_buttons = self._get_word_html_elements()
        self.all_guesses, self.previous_guesses = [], []
        self.attempts, self.total_turns = 4, 0

    def _get_word_html_elements(self) -> List[str]:
        """call after each submission because DOM refreshes each submit"""
        return {
            e.text: e
            for e in self.driver.find_elements(By.CLASS_NAME, S.ITEM_CLASS.value)
        }

    def _get_correct_groups(self) -> List[str]:
        """get previously submitted word groups that are correct"""
        correct_groups = []
        if self.driver.find_elements(By.CLASS_NAME, S.CORRECT_CLASS.value):
            correct_groups = [
                e.text
                for e in self.driver.find_elements(By.CLASS_NAME, S.CORRECT_CLASS.value)
            ]
        return correct_groups

    def turn(self) -> GameState:
        """take one turn of connections game"""
        print(f"turn: {self.total_turns}")
        # check if only one group left
        if self.check_win():
            self.submit_group(words)
            return GameState.GAMEWIN
        # make and submit guess
        words = self.llm_input_ouput(
            prompt_factory(self.word_to_buttons.keys(), self.previous_guesses)
        )

        # update game state based on correctness
        if self.check_guess(words):
            self.previous_guesses = []
            self.all_guesses.append(
                {"words": words, "correct": True, "turn": self.total_turns}
            )
            print("correct")
        else:
            self.previous_guesses.append(words)
            self.all_guesses.append(
                {"words": words, "correct": False, "turn": self.total_turns}
            )
            print("incorrect")

        self.total_turns += 1
        # update new bindings, clear board
        self.word_to_buttons = self._get_word_html_elements()
        self.deselect_words()
        if self.attempts_left() == 1:
            return GameState.GAMEOVER
        time.sleep(2)
        return GameState.PLAYING

    def request(self, prompt: str) -> str:
        """make request to openai gpt-4"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=500,
        )

        if "error" in response.json():
            print(response.json())
            raise ApiRequestError("Error while making the api request")

        return response.json()["choices"][0]["message"]["content"]

    @log_args_retval
    def llm_input_ouput(self, prompt: str) -> Tuple[str, str]:
        """
        serve prompt as input to llm, take llm output and submit to game
        return original list of word llm guessed
        """
        guess = self.request(prompt)
        words = [word.strip().upper() for word in guess[1:-1].split(",")]
        self.submit_group(words)
        return words

    def submit_group(self, words: List[str]) -> None:
        """submit a 4-letter list of words as grouping guess"""
        for word in words:
            word_button = self.word_to_buttons[word]
            self.driver.execute_script(
                f"""
                let pointerDown = new Event('pointerdown');
                let wordButton = document.getElementById('{word_button.get_attribute('id')}');
                wordButton.dispatchEvent(pointerDown);
                wordButton.classList.add('selected');
                """,
                word_button,
            )
            time.sleep(0.25)
        # submit guess; submit is span element -> dispatch pointerdown event
        self.driver.execute_script(
            """
            let pointerDown = new Event('pointerdown');
            let submitButton = document.getElementById('submit-button');
            submitButton.dispatchEvent(pointerDown);
            """
        )
        time.sleep(2)

    def check_guess(self, words: List[str]) -> bool:
        """check if guess is correct"""
        correct_groups = self._get_correct_groups()
        return len(set(words) - set(correct_groups)) == 0

    def check_win(self) -> bool:
        """check if only on group left, thus game is done (last guess is trivial)"""
        all_words = self.word_to_buttons.keys()
        correct_groups = self._get_correct_groups()
        words = list(set(all_words) - set(correct_groups))
        return len(words) == 4

    def attempts_left(self) -> int:
        """return number of connection attempts left"""
        attempts = self.driver.find_elements(By.CLASS_NAME, S.ATTEMPT_CLASS.value)
        self.attempts = len(
            list(
                filter(
                    lambda x: "lost" not in x,
                    [attempt.get_attribute("class") for attempt in attempts],
                )
            )
        )
        return self.attempts

    def deselect_words(self) -> None:
        """deselect previous guesses"""
        for word in self.word_to_buttons.values():
            if "selected" in word.get_attribute("class"):
                # deselect and update css class
                self.driver.execute_script(
                    f"""
                    let pointerCancel = new Event('pointercancel');
                    let wordButton = document.getElementById('{word.get_attribute('id')}');
                    wordButton.dispatchEvent(pointerCancel);
                    wordButton.classList.remove('selected');
                    """,
                    word,
                )
