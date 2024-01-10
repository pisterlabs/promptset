import json
import os
import openai
import unicodedata


class Tutor:
    _GPT_MODEL = "gpt-4"
    _STATS_PATH = './data/stats.json'
    _KANJI_PATH = './data/kanji_freq_list.txt'
    _TEMPERATURE = 0.7

    def __init__(self, openai_key):
        openai.api_key = openai_key
        self.messages = []
        self.lesson_stats = self.load_lesson_stats()

        # Load initial prompt
        with open("prompts/proposer_prompt.txt", 'r', encoding="utf-8") as f:
            self.messages.append({"role": "system", "content": f.read().strip()})

        self.messages.append({"role": "system", "content": f"User stats:\n{json.dumps(self.lesson_stats)}"})

    def speak(self, message: str, speaker="student") -> list[list]:
        self.messages.append({"role": "user", "content": message, "name": speaker})

        response = openai.ChatCompletion.create(
            model=self._GPT_MODEL,
            messages=self.messages,
            temperature=self._TEMPERATURE,
        )

        self.messages.append(response.choices[0]["message"])
        json_string = response.choices[0]["message"]["content"].strip()
        print(json_string)
        try:
            response_json = json.loads(json_string)
        except Exception as e:
            print(f"Failed to parse json string:\n{json_string}")
            raise e

        response_json = self.update_lesson_stats(response_json)
        return response_json

    def update_lesson_stats(self, response):
        mistake = len(response) > 0 and response[0][0].strip() == "Incorrect"
        if mistake:
            print("Mistake detected")
            # Remove mistake flag
            response = response[1:]

        for key, _ in response:
            self.inc_stats(key, 0)
            if mistake:
                self.inc_stats(key, 1)
        self.save_lesson_stats()
        return response

    def start_lesson(self):
        return self.speak("You may start the lesson.", speaker="app")

    def load_lesson_stats(self):
        stats = {}

        if os.path.exists(self._STATS_PATH):
            with open(self._STATS_PATH, "r") as file:
                stats = json.load(file)
        else:
            with open(self._STATS_PATH, "w") as file:
                json.dump(stats, file)

        print(stats)
        return stats

    def save_lesson_stats(self):
        with open(self._STATS_PATH, "w") as file:
            json.dump(self.lesson_stats, file)

    def extract_japanese_characters(self, sentence):
        japanese_characters = set()

        for char in sentence:
            char_name = unicodedata.name(char, '')

            if 'HIRAGANA' in char_name or 'KATAKANA' in char_name or 'CJK UNIFIED IDEOGRAPH' in char_name:
                japanese_characters.add(char)

        return japanese_characters

    def inc_stats(self, sentence, stat_idx):
        jap_chars = self.extract_japanese_characters(sentence)

        for c in jap_chars:
            if c not in self.lesson_stats:
                self.lesson_stats[c] = [0, 0]

            self.lesson_stats[c][stat_idx] += 1
