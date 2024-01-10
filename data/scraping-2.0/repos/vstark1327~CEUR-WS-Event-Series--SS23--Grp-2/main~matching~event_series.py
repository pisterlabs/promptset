import os
import openai
import json
from sklearn.model_selection import train_test_split
import copy
import re


class OpenAISeriesMatcher:
    def __init__(self):
        self.events = None
        self.series = None
        self.events_with_title_and_series = []
        self.event_series_with_title = []
        self.events_with_title_and_series_bijective = []
        self.train = []
        self.test = []
        self.test_dropped = []
        self.deduced_event_series = []
        self.extracted_texts = []

        openai.api_key = os.environ.get("OPENAI_API_KEY")

        self.read_files()
        self.preprocess()
        self.generate_prompt()
        self.request_save_response()
        self.extract_series_titles()

    def read_files(self):
        resources_path = os.path.abspath("resources")
        events = open(
            os.path.join(resources_path, "events.json"), "r", encoding="utf-8"
        )

        self.events = json.load(events)

        events.close()

        series = open(
            os.path.join(resources_path, "event_series.json"), "r", encoding="utf-8"
        )

        self.series = json.load(series)

        series.close()

    def preprocess(self):
        for binding in self.series["results"]["bindings"]:
            if "title" in binding:
                self.event_series_with_title.append(binding)

        for binding in self.events["results"]["bindings"]:
            if "title" in binding and "series" in binding:
                self.events_with_title_and_series.append(binding)

        event_series_dummy = [event["series"] for event in self.event_series_with_title]

        self.events_with_title_and_series_bijective = [
            event
            for event in self.events_with_title_and_series
            if event["series"] in event_series_dummy
        ]

        self.train, self.test = train_test_split(
            self.events_with_title_and_series_bijective, test_size=0.1
        )

        self.test_dropped = copy.deepcopy(self.test)

        for item in self.test_dropped:
            if "series" in item:
                del item["series"]

            if "seriesLabel" in item:
                del item["seriesLabel"]

            if "ordinal" in item:
                del item["ordinal"]

    def generate_prompt(self):
        self.conversation = [
            {"role": "system", "content": "You are a human"},
            {
                "role": "user",
                "content": 'In wikidata, there are about 3400 entries which are interesting to me. Lets call these as "events". Additionally, there are different and fewer group of entries, lets call these as "event series". Almost all events are a part of event series. I will provide real examples later on, but for context, we can draw similarities to this example: if each star wars movies are "events", then the star wars itself is the "event series".',
            },
            {
                "role": "user",
                "content": "In wikidata, the property connecting the events to event series are missing, and my task is to deduct from the title of the event which event series does this event belong to. For humans it is an easy task for sure, but noone wants to edit thousands of entries by hand. This is where you step in.",
            },
            {
                "role": "user",
                "content": "I want you to deduct which event series does the following events belong to. To help you out, i will provide titles of some random events, and their corresponding event series to help you out with the pattern recognition. Then i will provide more events for you to find out the event series for these.",
            },
        ]

        # feed training events & corresponding event series into conversation

        for count, item in enumerate(self.train, start=1):
            self.conversation.append(
                {
                    "role": "user",
                    "content": "Event "
                    + str(count)
                    + " is named '"
                    + item["title"]["value"]
                    + "'",
                }
            )
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": "The event series for Event "
                    + str(count)
                    + " is '"
                    + item["seriesLabel"]["value"]
                    + "'",
                }
            )

    def request_save_response(self):
        # feed test dataset titles
        for event in self.test_dropped:
            self.conversation.append(
                {
                    "role": "user",
                    "content": "The event for you to find event series is "
                    + event["title"]["value"],
                }
            )

            # Send the conversation to ChatGPT
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=self.conversation,
                stop=None,
                temperature=0.7,
            )

            # Extract the deduced event series from the response
            self.deduced_event_series.append(
                response["choices"][0]["message"]["content"]
            )

            # Add the assistant's response to the conversation history
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": response["choices"][0]["message"]["content"],
                }
            )

        resources_path = os.path.abspath("results")
        file1 = open(
            os.path.join(
                resources_path, "event_series_response_" + response["id"] + ".json"
            ),
            "w",
            encoding="utf-8",
        )
        json.dump(self.deduced_event_series, file1, indent=6)
        file1.close()

        file2 = open(
            os.path.join(
                resources_path, "whole_conversation_" + response["id"] + ".json"
            ),
            "w",
            encoding="utf-8",
        )
        json.dump(self.conversation, file2, indent=6)
        file2.close()

    def extract_series_titles(self):
        pattern = "'(.*?)'"
        self.extracted_texts = [
            re.search(pattern, s).group(1) if re.search(pattern, s) else ""
            for s in self.deduced_event_series
        ]

    def compare_results(self):
        sum = 0

        for line, org in zip(self.extracted_texts, self.test):
            if line == org["seriesLabel"]["value"]:
                sum += 1

        print("accuracy: ", sum / len(self.test))
