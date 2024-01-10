import openai
from dotenv import load_dotenv
import os
import re
import json
from datetime import datetime

load_dotenv()


class StoryCraft_AI:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.conversation_history = []

    def set_last_story(self, last_story):
        self.last_story = last_story

    @staticmethod
    def read_json_file(file_name):
        try:
            with open(file_name, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return None

    @staticmethod
    def write_json_file(file_name, data):
        with open(file_name, "w") as file:
            json.dump(data, file, indent=4)

    def save_story(self, author, content):
        data = self.read_json_file("stories.json") or {"stories": []}
        new_story = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "author": author,
            "content": content,
        }
        data["stories"].append(new_story)
        self.write_json_file("stories.json", data)

    @staticmethod
    def read_stories():
        data = StoryCraft_AI.read_json_file("stories.json")
        return data["stories"] if data else []

    def save_conversation_history(self):
        self.write_json_file("conversation_history.json", self.conversation_history)

    def read_conversation_history(self):
        return self.read_json_file("conversation_history.json")

    @staticmethod
    def remove_special_characters(input_string):
        pattern = r"[^a-zA-Z0-9\s]"
        return re.sub(pattern, "", input_string)

    def get_adjustment_params(self, adjustment_params, key, default_value):
        return (
            adjustment_params.get(key, default_value)
            if adjustment_params
            else default_value
        )

    def get_openai_response(self, messages, **kwargs):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4", messages=messages, **kwargs
            )
            return response["choices"][0]["message"]["content"]
        except openai.error.AuthenticationError:
            return "AuthenticationError: Please check your OpenAI API credentials."

    def session_to_story_gpt4(self, messages, adjustment_params=None):
        temperature = self.get_adjustment_params(adjustment_params, "temperature", 0.3)
        frequency_penalty = self.get_adjustment_params(
            adjustment_params, "frequency_penalty", 0.5
        )
        presence_penalty = self.get_adjustment_params(
            adjustment_params, "presence_penalty", 0.5
        )

        conversation_setup = self.get_conversation_setup(messages)

        self.conversation_history.extend(conversation_setup)
        response = self.get_openai_response(
            self.conversation_history,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        self.conversation_history.append({"role": "assistant", "content": response})
        self.save_conversation_history()
        return response

    @staticmethod
    def get_conversation_setup(messages):
        return [
            {
                "role": "system",
                "content": "You are an assistant trained to convert short DND session notes into high fantasy narratives.",
            },
            {
                "role": "user",
                "content": f"Convert the following DND session notes into a high fantasy narrative:\n {messages}",
            },
            {
                "role": "assistant",
                "content": "The party members are Seeker (automaton fighter), Asinis (human cleric), Astrea (druid), Serath (hollowed one fighter), and Yfo (satyr Bard).",
            },
            {
                "role": "user",
                "content": "Make the story elaborate, but do not add major details which were not in the session notes.",
            },
        ]

    def edit_story(self, last_story, edit_prompt):
        new_prompt = [
            {
                "role": "user",
                "content": f"Edit the following story to {edit_prompt}: {last_story}",
            }
        ]
        return self.get_openai_response(new_prompt)

    def summarize_story_title(self, last_story, summary_prompt):
        summary_prompt = [
            {
                "role": "user",
                "content": f"Summarize the following story into one line: {last_story}",
            }
        ]
        return self.get_openai_response(summary_prompt)
