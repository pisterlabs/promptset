from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from furhat_remote_api import FurhatRemoteAPI  # type: ignore
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pkg_resources import resource_filename

from botender.interaction import gestures
from botender.interaction.drink_recommendation import DrinkRecommender
from botender.interaction.gaze_coordinator import GazeClasses, GazeCoordinatorThread
from botender.perception.detectors.speech_detector import SpeechDetector
from botender.perception.perception_manager import PerceptionManager
from botender.webcam_processor import WebcamProcessor

logger = logging.getLogger(__name__)

DRINKS_DATA_PATH = resource_filename(
    __name__, "drinks/drinks_with_categories_and_ranks.csv"
)


def get_openai_response(messages: list[ChatCompletionMessageParam]) -> str:
    """Returns the response from OpenAI API"""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    if (answer := response.choices[0].message.content) is None:
        raise ValueError("OpenAI API returned None")
    logger.debug(f"OpenAI API response: {answer}")
    return answer


def get_valence_from_message(message: str) -> Literal["Positive"] | Literal["Negative"]:
    """Returns the valence from the message"""

    if os.getenv("ENABLE_OPENAI_API") != "True":
        return "Positive"

    chat_messages = [
        {
            "role": "system",
            "content": """You are an endpoint.
                You will receive a text that a user said upon asking a question.
                The question is some kind of yes or no question. The user's response can therefore be either positive or negative.

                From the user input you will extract whether the user answered the question in a positive or negative manner.

                If the user sounds uncertain or asks a counterquestion, it usually means that the response is negative.

                If the user doesn't say anything that can be interpreted as a response, return Error.

                The response should have the following structure:
                [Positive | Negative | Error]

                Examples:
                Input: "Ah yeah this drink sounds tasty!"
                Your response: Positive

                Input: "Maybe another time."
                Your response: Negative""",
        },
        {
            "role": "user",
            "content": message,
        },
    ]
    try:
        valence = get_openai_response(chat_messages)  # type: ignore[arg-type]
        if valence == "Error":
            raise ValueError("OpenAI API returned Error")
        if valence not in ["Positive", "Negative"]:
            raise ValueError("OpenAI API returned invalid valence")
    except ValueError as e:
        raise ValueError from e

    return valence  # type: ignore[return-value]


class InteractionCoordinator:
    """The InteractionCoordinator is responsible for coordinating the interaction and starting
    gaze coordination thread.

    It follows the `state` design pattern. The `InteractionCoordinator` is the context.
    More info here: https://refactoring.guru/design-patterns/state"""

    _perception_manager: PerceptionManager
    _webcam_processor: WebcamProcessor
    _speech_detector: SpeechDetector
    _gaze_coordinator: GazeCoordinatorThread
    _furhat: FurhatRemoteAPI
    _recommender: DrinkRecommender
    _state: InteractionState
    _previous_state: InteractionState | None
    _last_emotion_detection: float = 0
    user_info: dict[str, str]

    def __init__(
        self,
        perception_manager: PerceptionManager,
        webcam_processor: WebcamProcessor,
        gaze_coordinator: GazeCoordinatorThread,
        furhat: FurhatRemoteAPI,
    ):
        self._perception_manager = (
            perception_manager  # Used to get results from perception subsystem
        )
        self._webcam_processor = webcam_processor  # Used to interact with GUI
        self._furhat = furhat  # Used to interact with Furhat
        self._gaze_coordinator = gaze_coordinator
        self._recommender = DrinkRecommender(DRINKS_DATA_PATH)
        self._speech_detector = SpeechDetector(self._furhat)
        self._state = None  # type: ignore[assignment]
        self.transition_to(GreetingState())  # Initial state
        self.user_info = {}

    def transition_to(self, state: InteractionState):
        """The Context allows changing the State object at runtime."""
        logger.info(f"Interaction state transitioned to {type(state).__name__}")
        self._previous_state = self._state
        self._state = state
        self._state.context = self

    def handle(self):
        """The Context delegates part of its behavior to the current State object."""
        self._state.handle()

    def listen(self) -> str:
        """Listens to the user and returns the text."""
        self._furhat.gesture(
            body=gestures.get_random_gesture("listening"), blocking=False
        )
        return self._speech_detector.capture_speech()

    def get_emotion(self) -> str:
        """Returns the emotion of the user."""
        # while self._perception_manager.detects_emotion():
        #     time.sleep(1)
        if self._perception_manager.current_result is None:
            return "neutral"
        return self._perception_manager.current_result.emotion

    def set_gaze(self, gaze_class: GazeClasses) -> None:
        """Sets the gaze to follow the user."""
        self._gaze_coordinator.set_gaze_state(gaze_class)

    def interact(self) -> InteractionCoordinator | None:
        """Runs one interaction cycle and returns the updated InteractionCoordinator or
        None if the interaction is finished."""

        self.handle()

        if isinstance(self._state, FarewellState):
            self.handle()  # One more time to say goodbye
            logger.info("Interaction finished.")
            return None

        # If no face is present, transition to search state
        if not self._perception_manager.face_present and not isinstance(
            self._state, SearchState
        ):
            timeout = 0.0
            while not self._perception_manager.face_present:
                logger.info("No face present.")
                time.sleep(0.5)
                timeout += 0.5
                if timeout > 3:
                    logger.info("Timeout reached.")
                    self.transition_to(SearchState())
                    break

        if time.time() - self._last_emotion_detection > 5:
            self._perception_manager.detect_emotion()
            self._last_emotion_detection = time.time()

        return self


class InteractionState(ABC):
    """Abstract class for all interaction states."""

    @property
    def context(self) -> InteractionCoordinator:
        return self._context

    @context.setter
    def context(self, context: InteractionCoordinator):
        self._context = context

    @abstractmethod
    def handle(self):
        """Handles the interaction."""
        ...


class GreetingState(InteractionState):
    """State to handle greeting the user"""

    GREETINGS: list[str] = [
        "Hello there! I am Botender, your friendly neighborhood robotic bartender.",
        "Hey hey, I am Botender, the robotic bartender.",
        "Greetings, soo good to see you! My name is Botender.",
        "Hello, I am Botender, your friendly robotic bartender.",
    ]

    def handle(self):
        furhat = self.context._furhat

        # Set the gaze to follow the user
        self.context.set_gaze(GazeClasses.FACE)

        # Greet the user
        furhat.gesture(name="Smile", blocking=False)
        greeting = self.GREETINGS[np.random.randint(0, len(self.GREETINGS))]

        # TODO Add gestures
        furhat.say(text=greeting, blocking=True)

        # Transition to introduction state
        self.context.transition_to(IntroductionState())


class IntroductionState(InteractionState):
    """State to handle introducing the user to the robot"""

    INTRODUCTION_QUESTIONS: list[str] = [
        "What is your name?",
        "And who might you be?",
        "What is your name, if I may ask?",
        "It's a pleasure to meet you! What is your name?",
    ]

    @staticmethod
    def get_name_from_message(message: str) -> str:
        """Returns the name from the message"""

        if os.getenv("ENABLE_OPENAI_API") != "True":
            return "Paul"

        chat_messages = [
            {
                "role": "system",
                "content": 'You are an endpoint.\nYou will receive a text that a user said upon asking for his/her name.\n\nLook for the name of the user in the text.\nIf you are certain about the name return it.\nIf you are uncertain, only return "Error".\n\nThe response should have the following structure:\n\n[NAME OR ERROR]\n\nExamples:\nInput: "Ah yeah so good to meet you how exciting I\'m John by the way"\n\nYour response:\nJohn\n\nInput: "I have never seen anything like you Botender."\n\nYour response:\nError',
            },
            {
                "role": "user",
                "content": message,
            },
        ]
        try:
            name = get_openai_response(chat_messages)  # type: ignore[arg-type]
            if name == "Error":
                raise ValueError("OpenAI API returned Error")
        except ValueError as e:
            raise ValueError from e

        return name

    def handle(self):
        furhat = self.context._furhat
        introduction_question = self.INTRODUCTION_QUESTIONS[
            np.random.randint(0, len(self.INTRODUCTION_QUESTIONS))
        ]
        furhat.gesture(name="Smile", blocking=False)
        furhat.say(text=introduction_question, blocking=True)

        # self._context._perception_manager.detect_emotion()
        user_response = self.context.listen()

        try:
            name = self.get_name_from_message(user_response)
        except ValueError:
            furhat.gesture(
                body=gestures.get_random_gesture("understand_issue"),
                blocking=False,
            )
            furhat.say(text="I'm sorry, I didn't quite get that.", blocking=True)
            return

        self.context.user_info["name"] = name

        furhat.gesture(name="Smile", blocking=False)
        furhat.say(
            text=f"Nice to meet you {self.context.user_info['name']}.", blocking=True
        )
        self.context.transition_to(AcknowledgeEmotionState())


class AcknowledgeEmotionState(InteractionState):
    """State to handle aknowledging the user's emotion"""

    ACKNOWLEDGE_EMOTION_TEXTS: list[str] = [
        "You seem {emotion}.",
        "You look {emotion}.",
        "You seem {emotion} today.",
        "You look {emotion} today.",
    ]

    def handle(self):
        furhat = self.context._furhat

        emotion = self.context.get_emotion()
        if emotion == "happy" or emotion == "neutral":
            furhat.gesture(body=gestures.get_random_gesture("happy"), blocking=False)
        elif emotion == "sad" or emotion == "angry":
            furhat.gesture(body=gestures.get_random_gesture("concern"), blocking=False)

        acknowledge_emotion_text = self.ACKNOWLEDGE_EMOTION_TEXTS[
            np.random.randint(0, len(self.ACKNOWLEDGE_EMOTION_TEXTS))
        ]

        # Evaluation of f-string was postponed to runtime
        acknowledge_emotion_text = eval(f"f'{acknowledge_emotion_text}'")

        furhat.say(text=acknowledge_emotion_text, blocking=True)
        self.context.transition_to(AskDrinkState())


class AskDrinkState(InteractionState):
    """State to start the drink recommendation flow"""

    DRINK_QUESTIONS = [
        "Can I interest you in a drink?",
        "Would you like a drink?",
        "How about a drink?",
        "Are you in the mood for a drink?",
    ]

    POSITIVE_VALENCE_RESPONSES = [
        "That's great!",
        "That's awesome!",
        "That's great to hear!",
        "That's great to know!",
    ]

    NEGATIVE_VALENCE_RESPONSES = [
        "I'm sorry to hear that.",
        "I'm sorry to hear that, maybe next time.",
        "I'm sorry to hear that, maybe next time you will be in the mood for a drink.",
        "Alright, just let me know if you change your mind.",
    ]

    def handle(self):
        furhat = self.context._furhat
        drink_question = self.DRINK_QUESTIONS[
            np.random.randint(0, len(self.DRINK_QUESTIONS))
        ]
        furhat.say(text=drink_question, blocking=True)

        user_response = self.context.listen()

        try:
            valence = get_valence_from_message(user_response)
        except ValueError:
            furhat.gesture(
                body=gestures.get_random_gesture("understand_issue"),
                blocking=False,
            )
            furhat.say(text="I'm sorry, I didn't quite get that.", blocking=True)
            return

        if valence == "Positive":
            furhat.gesture(body=gestures.get_random_gesture("happy"), blocking=False)
            positive_valence_response = self.POSITIVE_VALENCE_RESPONSES[
                np.random.randint(0, len(self.POSITIVE_VALENCE_RESPONSES))
            ]
            furhat.say(text=positive_valence_response, blocking=True)
            self.context.transition_to(AskTastePreference())
        elif valence == "Negative":
            furhat.gesture(body=gestures.get_random_gesture("concern"), blocking=False)
            negative_valence_response = self.NEGATIVE_VALENCE_RESPONSES[
                np.random.randint(0, len(self.NEGATIVE_VALENCE_RESPONSES))
            ]
            furhat.say(text=negative_valence_response, blocking=True)
            self.context.transition_to(FarewellState())


class AskTastePreference(InteractionState):
    TASTE_PREFERENCE_QUESTIONS = [
        "What kind of cocktail do you like? I have sweet, milk-based, sour, and strong cocktails.",
        "My cocktails are sour, sweet, milk-based, or strong. What do you prefer?",
    ]

    TASTE = Literal["Sour", "Sweet", "Milk-based", "Strong"]

    def get_taste_preference_from_message(self, message: str) -> TASTE:
        """Returns the taste preference from the message"""

        if os.getenv("ENABLE_OPENAI_API") != "True":
            return "Sour"

        chat_messages = [
            {
                "role": "system",
                "content": """You are an endpoint.
                    You will receive a text that a user said upon asking for his/her taste preference.
                    Available tastes are
                    Sweet, Sour, Milk-based, and Strong.

                    Look for the taste preference of the user in the text.

                    If you are certain about the taste preference return it.
                    You can also infer a taste from the message, if it fits. For example, "I have had a rough day" requires something strong and "I'm a real sweet-tooth" probably means the user wants something sweet.

                    If it is completely uncertain, return Error.

                    The response should have the following structure:
                    [Sweet | Sour | Milk-based | Strong | Error]

                    Examples:

                    Input: "I like sweet cocktails."
                    Your response: Sweet

                    Input: "I want something heavy with a lot of alcohol."
                    Your response: Strong""",
            },
            {
                "role": "user",
                "content": message,
            },
        ]
        try:
            taste_preference = get_openai_response(chat_messages)  # type: ignore[arg-type]
            if taste_preference == "Error":
                raise ValueError("OpenAI API returned Error")
            if taste_preference not in ["Sour", "Sweet", "Milk-based", "Strong"]:
                raise ValueError("OpenAI API returned invalid taste preference")
        except ValueError as e:
            raise ValueError from e

        return taste_preference  # type: ignore[return-value]

    def handle(self):
        furhat = self.context._furhat

        taste_preference_question = self.TASTE_PREFERENCE_QUESTIONS[
            np.random.randint(0, len(self.TASTE_PREFERENCE_QUESTIONS))
        ]
        furhat.say(text=taste_preference_question, blocking=True)

        user_response = self.context.listen()

        try:
            taste_preference = self.get_taste_preference_from_message(user_response)
        except ValueError:
            furhat.gesture(
                body=gestures.get_random_gesture("understand_issue"),
                blocking=False,
            )
            furhat.say(text="I'm sorry, I didn't quite get that.", blocking=True)
            return

        self.context.user_info["taste_preference"] = taste_preference

        self.context.transition_to(RecommendDrinksState())


class RecommendDrinksState(InteractionState):
    """State to start the drink recommendation flow"""

    def generate_cocktail_description(self, cocktail_name: str, ingredients: str):
        """Returns a cocktail description based on the ingredients"""

        default_description = f"I can recommend you a {cocktail_name}."

        if os.getenv("ENABLE_OPENAI_API") != "True":
            return default_description

        chat_messages = [
            {
                "role": "system",
                "content": """You are an endpoint.
                    You will receive a cocktail name along with its ingredients.
                    Your task is to generate an enticing but short description of the
                    cocktail in the following format: I can recommend you a "cocktailname".
                    It is _ Examples: Input:  KING OF KINGSTON,"1 ounce gin, 1 teaspoon grapefruit, 0.5 ounce creme de,
                    1 teaspoon grenadine, 1 ounce pineapple juice1 ounce heavy cream, 1 ounce dark rum,
                    1 ounce light rum, Â½ ounce cherry brandy 1 pineapple slice, 4 ounces pineapple juiceYour response:
                    I can recommend you a King of Kingston. It is a delightful mix with a high sweetness score,
                    combining the unique flavors of grapefruit and pineapple with a touch of creamy crème de cacao.""",
            },
            {
                "role": "user",
                "content": f'name: "{cocktail_name}" , ingredients: {ingredients}',
            },
        ]
        try:
            cocktail_description = get_openai_response(chat_messages)  # type: ignore[arg-type]
            if cocktail_description == "Error":
                raise ValueError("OpenAI API returned Error")
        except ValueError:
            return default_description

        return cocktail_description  # type: ignore[return-value]

    def handle(self):
        recommender = self.context._recommender
        furhat = self.context._furhat
        emotion = self.context.get_emotion()

        taste_preference = self.context.user_info["taste_preference"]

        if taste_preference not in ["Sour", "Sweet", "Milk-based", "Strong"]:
            self.context.transition_to(AskTastePreference())
            return

        # Call the recommend_drink method to get a recommendation
        random_recommendation = recommender.recommend_drink(emotion, taste_preference)

        # Since the recommendation is a DataFrame, extract the first row
        # and access the 'Cocktail' and 'Ingredients' columns
        if not random_recommendation.empty:
            cocktail_name = random_recommendation["Cocktail"].iloc[0]
            self.context.user_info["cocktail_name"] = cocktail_name
            ingredients = random_recommendation["Ingredients"].iloc[0]

            furhat.gesture(body=gestures.get_random_gesture("thinking"), blocking=False)
            cocktail_description = self.generate_cocktail_description(
                cocktail_name, ingredients
            )
            furhat.say(text=cocktail_description, blocking=True)

            self.context.transition_to(FeedbackForDrinkRecommendationState())
        else:
            furhat.gesture(
                body=gestures.get_random_gesture("concern"),
                blocking=False,
            )
            furhat.say(
                text="I'm sorry, but no recommendation is available for the given criteria.",
                blocking=True,
            )
            self.context.transition_to(AskTastePreference())
            return


class FeedbackForDrinkRecommendationState(InteractionState):
    """State to handle feedback for drink recommendation"""

    FEEDBACK_QUESTIONS = [
        "Does that sound good to you?",
        "Would you like to try that?",
        "Does that sound like something you would enjoy?",
        "Would you like to try that cocktail?",
    ]

    def handle(self):
        furhat = self.context._furhat

        feedback_question = self.FEEDBACK_QUESTIONS[
            np.random.randint(0, len(self.FEEDBACK_QUESTIONS))
        ]

        furhat.say(
            text=feedback_question,
            blocking=True,
        )

        user_response = self.context.listen()

        try:
            valence = get_valence_from_message(user_response)
        except ValueError:
            furhat.gesture(
                body=gestures.get_random_gesture("understand_issue"),
                blocking=False,
            )
            furhat.say(text="I'm sorry, I didn't quite get that.", blocking=True)
            return

        if valence == "Positive":
            cocktail_name = self.context.user_info["cocktail_name"]
            furhat.gesture(body=gestures.get_random_gesture("happy"), blocking=False)
            furhat.say(
                text=f"That's great! Here is your {cocktail_name}.", blocking=True
            )
            self.context.transition_to(FarewellState())
        elif valence == "Negative":
            furhat.gesture(body=gestures.get_random_gesture("concern"), blocking=False)
            furhat.say(text="Alright, I will look up another cocktail.", blocking=True)
            self.context.transition_to(RecommendDrinksState())


class FarewellState(InteractionState):
    """State to handle saying goodbye to the user"""

    FAREWELL_TEXTS = [
        "It was a pleasure serving you! Goodbye!",
        "I hope to see you again soon! Bye!",
        "I hope you have a good day! Bye!",
    ]

    def handle(self):
        furhat = self.context._furhat
        furhat.gesture(body=gestures.get_random_gesture("happy"), blocking=False)
        farewell_text = self.FAREWELL_TEXTS[
            np.random.randint(0, len(self.FAREWELL_TEXTS))
        ]
        furhat.say(text=farewell_text, blocking=True)

        time.sleep(10)


class SearchState(InteractionState):
    """State to handle looking for the user"""

    SEARCH_QUESTIONS = [
        "Where did you go?",
        "Where did you go? I can't see you.",
        "Are you still there?",
        "Are you still there? I can't see you anymore.",
        "Where are you?",
    ]

    def handle(self):
        furhat = self.context._furhat
        question_count = 0
        while not self.context._perception_manager.face_present:
            furhat.gesture(
                body=gestures.get_random_gesture("understand_issue"), blocking=False
            )
            search_question = self.SEARCH_QUESTIONS[
                np.random.randint(0, len(self.SEARCH_QUESTIONS))
            ]
            furhat.say(text=search_question, blocking=True)
            question_count += 1
            time.sleep(2)
            if question_count >= 2:
                furhat.gesture(
                    body=gestures.get_random_gesture("concern"),
                    blocking=False,
                )
                furhat.say(
                    text="I'm sorry, I can't find you. I will look for you later.",
                    blocking=True,
                )
                self.context.transition_to(FarewellState())
                return

        furhat.gesture(body=gestures.get_random_gesture("happy"), blocking=False)
        furhat.say(text="Oh, there you are!", blocking=True)
        self.context.transition_to(self.context._previous_state)
