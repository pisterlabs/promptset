"""Fully integrated language assistant"""
import time
from typing import Dict, List, Optional

import questionary
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel

from languageassistant.agents.conversation import (
    BaseConversationAgent,
    load_conversation_agent,
)
from languageassistant.agents.planner import (
    BasePlannerAgent,
    Lesson,
    load_lesson_planner,
)
from languageassistant.transcriber import Transcriber
from languageassistant.tts import TTS


class Assistant(BaseModel):
    """Full language assistant model"""

    language: str
    """Target language"""
    proficiency: str
    """User proficiency with target language"""
    lesson: Lesson = Lesson(topics=[])
    """Lesson plan of topics to discuss"""

    llm: BaseLanguageModel = ChatOpenAI(temperature=0)  # type: ignore[call-arg]
    """Language model for inference"""
    lesson_agent: BasePlannerAgent = load_lesson_planner(llm)
    """Agent to use for planning lessons"""
    conversation_agent: BaseConversationAgent = load_conversation_agent(llm)
    """Agent to use for conversing about a topic"""
    transcriber: Transcriber
    """Voice transcription model"""
    tts: TTS
    """Text-to-speech model"""
    use_tts: bool = True
    """If TTS should be used or not"""

    class Config:
        arbitrary_types_allowed = True

    @property
    def background(self) -> Dict[str, str]:
        """User target language background"""
        return {"language": self.language, "proficiency": self.proficiency}

    def plan_lesson(self) -> None:
        """Plan a lesson for the target language using user's background"""
        self.lesson = self.lesson_agent.plan(self.background)

    def greet(self, topic: str) -> str:
        """Receive background teaching about a topic"""
        inputs = self.background.copy()
        inputs["topic"] = topic
        return self.conversation_agent.greet(inputs)

    def speak(self, topic: str, human_input: str) -> str:
        """Single response from user's conversation input about a topic"""
        inputs = self.background.copy()
        inputs["topic"] = topic
        inputs["human_input"] = human_input
        return self.conversation_agent.speak(inputs)

    def _converse(self, topic: str) -> None:
        """Private converse method to reduce indentation in actual converse method"""
        # Have Assistant start the conversation
        ai_response = self.speak(topic, "What should I know first?")
        print("\rAssistant:", ai_response)
        if self.use_tts:
            self.tts.run(ai_response)

        # Clear transcription recording buffer before conversation
        self.transcriber.clear_recording_buffer()

        # Begin conversation loop
        while True:
            # Get user input
            print("Microphone recording...", end="")
            user_input = self.transcriber.run()

            # User did not say anything
            if user_input == "":
                print("\r", end="")
                time.sleep(0.25)
                continue

            # Overwrite microphone recording message
            r_padding = " " * max(23 - 7 - len(user_input), 0)
            print("\rHuman:", user_input, r_padding)

            # Retrieve Assistant response
            print("Retrieving response...", end="")
            ai_response = self.speak(topic, user_input)

            # Assistant detects user wishes to end the conversation
            if "<END_CONVERSATION>" in ai_response:
                return

            # Output Assistant response
            print("\rAssistant:", ai_response)
            if self.use_tts:
                self.tts.run(ai_response)

            self.transcriber.clear_recording_buffer()

    def converse(self, topic: str) -> None:
        """Converse with conversation agent about a topic"""
        try:
            self._converse(topic)
        except KeyboardInterrupt:
            return

    def _output_topic_background(self, topic: str) -> None:
        """Output topic background teaching"""
        print("Retrieving topic background teaching...")
        topic_prereqs = self.greet(topic)
        print(topic_prereqs)
        if self.use_tts:
            self.tts.run(topic_prereqs)

    def run(
        self, include_topic_background: bool, lesson: Optional[List[str]] = None
    ) -> None:
        """
        Full assistant lesson planning, background teaching, and conversation loop

        Parameters
        ----------
        include_topic_background
            If the user wants background teaching before starting the conversation
        lesson
            An optional custom lesson plan of topics to discuss
        """
        # Get lesson plan if not provided
        if lesson is None:
            self.plan_lesson()
        else:
            self.lesson = Lesson(topics=lesson)

        # Display topics
        print("List of conversation topics:")
        print(self.lesson)

        # TODO: Allow user to provide feedback to adjust topics (too easy, too hard, done before)
        # Can probably use regular LLM for this?

        # Begin lesson
        for topic in self.lesson.topics:
            print("Starting new conversation. Press CTRL + C to end conversation")
            print("Topic:", topic)

            # Clear transcription recording buffer before conversation
            self.transcriber.clear_recording_buffer()

            # Begin conversation
            if include_topic_background:
                self._output_topic_background(topic)
            self.converse(topic)

            # Continue to next topic or return
            if topic == self.lesson.topics[-1]:  # No more topics -> return
                return
            continue_conversations = questionary.select(
                "Conversation ended. Continue to next topic?",
                choices=[
                    "Yes",
                    "No",
                ],
            ).ask()
            if continue_conversations == "No":
                return
