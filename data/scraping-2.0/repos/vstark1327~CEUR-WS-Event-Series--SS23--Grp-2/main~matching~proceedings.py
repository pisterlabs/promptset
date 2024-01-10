import os
import openai
import json
import re
import random


class OpenAIProceedingsMatcher:
    def __init__(self):
        self.events = None
        self.series = None
        self.proceedings = None
        self.events_with_title_and_series = []
        self.event_series_with_title = []
        self.events_with_title_and_series_bijective = []
        self.deduced_event_series = []
        self.proceedings_with_title = []
        self.proceedings_sample = []
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

        proceedings = open(
            os.path.join(resources_path, "proceedings.json"), "r", encoding="utf-8"
        )
        self.proceedings = json.load(proceedings)
        proceedings.close()

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

        for binding in self.proceedings["results"]["bindings"]:
            if "proceedingTitle" in binding:
                self.proceedings_with_title.append(binding)

        self.proceedings_sample = random.sample(self.proceedings_with_title, 25)

    def generate_prompt(self):
        self.conversation = [
            {"role": "system", "content": "You are a human"},
            {
                "role": "user",
                "content": 'In wikidata, there are about 3400 entries which are interesting to me. Lets call these as "events". Additionally, there are different and fewer group of entries, lets call these as "event series". Almost all events are a part of event series. I will provide real examples later on, but for context, we can draw similarities to this example: if each star wars movies are "events", then the star wars itself is the "event series".',
            },
            {
                "role": "user",
                "content": 'Moreover, there are entities called "proceedings". "proceedings" are collection of scientific papers which are presented in the earlier mentioned "events". It is known that every proceeding comes from an event, or multiple events can publish their proceedings jointly.',
            },
            {
                "role": "user",
                "content": "In wikidata, the property connecting the events to event series are missing, and my task is to deduct from the title of the proceedings which event series does this event belong to. For humans it is an easy task for sure, but noone wants to edit thousands of entries by hand. This is where you step in.",
            },
            {
                "role": "user",
                "content": "I want you to deduct which event series does the following proceedings belong to. To help you out, i will provide titles of some random events, and their corresponding event series to help you out with the pattern recognition. Then i will provide some sample of proceedings title for you to find out the event series for these.",
            },
        ]

        # feed all events & corresponding event series into conversation

        for count, item in enumerate(
            self.events_with_title_and_series_bijective, start=1
        ):
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
        for proceeding in self.proceedings_sample:
            self.conversation.append(
                {
                    "role": "user",
                    "content": "The title of 'proceeding' for you to find event series is "
                    + ""
                    + proceeding["proceedingTitle"]["value"]
                    + "",
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
                resources_path, "proceedings_response_" + response["id"] + ".json"
            ),
            "w",
            encoding="utf-8",
        )
        json.dump(self.deduced_event_series, file1, indent=6)
        file1.close()

        file2 = open(
            os.path.join(
                resources_path,
                "proceedings_whole_conversation_" + response["id"] + ".json",
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
