import random

import openai

from hangman.gallows import gallows
from hangman.words import words

openai_comp_prompt = """Give a hint of 7 words or less for the word '{hidden_word}' in Hangman using an unusual wording of the hint that will make the player genuinely interested in guessing the word and does not contain the word, avoid common or obvious associations with the word.

The hint should be in English and have a medium difficulty level.

- Example 1: Hidden word: assembler
  Hint: a low-level mnemonic programming language
- Example 2: Hidden word: education
  Hint: is the process of facilitating learning
- Example 3: Hidden word: media
  Hint: newspapers, facebook, instagram, ...
- Example 4: Hidden word: privacy
  Hint: keeping your personal information data safe
- Example 5: Hidden word: field
  Hint: a single element of data
- Example 6: Hidden word: pedagogy
  Hint: a model of teaching and learning

Response format:
Hint: [Write here]
"""


class Hangman:
    def __init__(self):
        self.lives_left = len(gallows) - 1

        self.hidden_word = random.choice(words).upper()
        self.unguessed_letters = set(self.hidden_word)
        self.used_letters = set()

        self.hint = 0
        try:
            self.hints = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=openai_comp_prompt.format(hidden_word=self.hidden_word),
                max_tokens=32,
                temperature=0.8,
                n=self.lives_left,
            )
        except:
            raise Exception("Failed to connect to OpenAI API.")

    def try_to_guess(self, player_letter):
        if player_letter in self.used_letters:
            return "You have already used this letter. Try again."
        else:
            self.used_letters.add(player_letter)
            if player_letter in self.unguessed_letters:
                self.unguessed_letters.remove(player_letter)
                return "You have guessed the new letter!"
            else:
                self.lives_left -= 1
                self.hint += 1
                return f"{player_letter} is not in the word."

    def game_goes_on(self):
        return len(self.unguessed_letters) > 0 and self.lives_left > 0

    def puzzle(self):
        return " ".join(
            [
                letter if letter in self.used_letters else "-"
                for letter in self.hidden_word
            ]
        )

    def get_hint(self):
        return self.hints.choices[self.hint].text.strip()
