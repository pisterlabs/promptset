import json

from utils.openai_client import OpenAIClient, get_openai_client
from utils.firebase_client import FirebaseClient
from utils.firestore_dao import FirestoreDAO
from schemas.story_components import *


class StoryService:

    def __init__(self, openai_client: OpenAIClient, firebase_client: FirebaseClient):
        self.openai_client = openai_client
        self.firebase_client = firebase_client
        self._take_action_rules = """
        We are playing a game similar to DND. 
        You will be the dungeon master, and I will be the player. 
        You will output a story_bit to give some story leading into the event.
        Then, you will output an event (prompt in the Json below).
        For each event, I will give you a unique response.
        Provide output only in Json.as

        context: {0}

        There are {1} events remaining

        Respond in this format:
        """
        self._take_action_json_output = """
        {
            "event": 
                {
                "story_bit": str (between 20 and 30 words)
                "event_title": str 
                "prompt": str (between 20 and 30 words and end with "What will you do?")
                }
            "summary":
                {
                "current_summary": str (key points of the previous context + story_bit + prompt, less than 60 words)
                }
            "character":
                {
                "health": int (increase or decrease after events)
                "items": (add or remove items as they are used)
                [
                {
                    "item_name": str
                    "item_type": str                
                }
                ]
                }
        }
        """
        self._initialize_story_rules = """
        Present me with an introduction to an adventure involving my character.
        Do all of this in between 60 and 70 words.
        Provide output only in Json.

        The json will look like this:
        """
        self._initialize_story_json_output = """
        {
            "initial_story": 
                {
                    "introduction": str (less than 50 words)
                }
        }
        """
        self._end_story_rules = """"""

    def initialize_story(self, character_id: str, story_length: StoryLengthEnum) -> InitialStory:
        character = self.firebase_client.get_character(character_id)

        character_description = (f"Character:"
                                 f"health: {character.get('health')}"
                                 f"character_class: {character.get('class')}")

        system_prompt = self._initialize_story_rules + self._initialize_story_json_output

        new_story_response = self.openai_client.generate_text(
            prompt=character_description,
            system_prompt=system_prompt,
            max_tokens=100)

        new_story_dict: dict[str, any] = json.loads(new_story_response.content)
        new_story_dict.update({
            "max_iterations": story_length.value,
            "character_id": character_id,
            "items": None,
            "id": None
        })
        print(new_story_dict)
        new_story = InitialStory(**new_story_dict)

        created_story = self.firebase_client.create_initial_story(new_story)

        return created_story

    def take_action(self, story_id: str, action: str) -> Event:
        max_tokens = 400
        current_iteration = 0
        last_action = False

        # retrieve story intro + events
        story = self.firebase_client.get_story(story_id)
        intro = story.get("initial_story").get("introduction")
        max_iterations = story.get("max_iterations")

        events = self.firebase_client.get_events(story_id)

        # if no events, then generate event with intro as context
        if len(events) == 0:
            context = intro
        # else use last event.current_summary as context
        else:
            context = events[0].get("summary").get("current_summary")
            current_iteration = events[0].get("iteration")
            # should check if iteration == to story.iteration
            if max_iterations >= current_iteration:
                last_action = True

        updated_rules = self._take_action_rules.format(context, max_iterations - current_iteration)
        system_prompt = updated_rules + self._take_action_json_output

        # if last_action:
        #     system_prompt = self._end_story_rules

        new_event_response = self.openai_client.generate_text(
            prompt=action,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )

        new_event_dict: dict[str, any] = json.loads(new_event_response.content)
        new_event_dict.update({
            "iteration": current_iteration + 1,
            "id": None
        })
        new_event = Event(**new_event_dict)
        created_event = self.firebase_client.add_event_to_story(story_id, new_event)
        return created_event


def get_story_service():
    openai_client = get_openai_client()
    firestore_dao = FirestoreDAO("firebase_creds.json")
    firebase_client = FirebaseClient(firestore_dao)
    return StoryService(openai_client, firebase_client)


if __name__ == '__main__':
    story_service = get_story_service()
    new_event = story_service.take_action("RahfZPCw0fdnZdXdl39w", "hello")
    print(new_event)
    # new_story = story_service.initialize_story("OTVnGkavbMuIvzj0DG1B", StoryLengthEnum.Short)
    # print(new_story)
