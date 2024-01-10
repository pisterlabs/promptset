import openai

import logging

from emobots.intention import intention_analysis
from emobots.mood import mood_analysis
from emobots.tools import message2string, strip_response


class Emobot:
    def __init__(self, client, name, person_desc) -> None:
        self.client = client
        self.name = name
        self._reccurrent_system_prompt = f"This is {name}: {person_desc}."

        self._mood_analyis_system_prompt = f"""
        From the above conversation, what do you think is the current mood of {name}?
        If uncertain, simply state that the mood is unknown."""

        self._intention_analyis_system_prompt = f"""
        From the above conversation, what does {name} wants to do from these options?
        1. keep the conversation going
        2. stop the converstation
        3. wait for the other person to say more
        4. accept the other persons offer

        Just give the number.
        If uncertain, simply state 'None'.
        """

        self._roleplay_system_prompt = f"""Imagine you are {self.name}. Write a response that {self.name}
        would write in a chat in a style and grammar that matches
        {self.name}'s background and proficiency with computers. Stay in character and tell only as much as the character would do.
        
        Keep in mind that people often are not aware of their character traits and have a hard time describing themselves.
        Also, they do not always speak in full sentences.

        Also, they do not always speak in full sentences.

        Also, people sometimes are lying or hiding things.

        
        {self.name} is chatting with the user only, often no full sentences and few words. Typing is hard work.
         
        Only give the response of {self.name} without any other text or responses of other people."""
        # Also, ommit the '{self.name}: ' or 'Other:' at the beginning of the response."""

        self.chat_messages = []

        self.current_feeling = "Neutral."

        self.intention = None

    def response_generator(
        self,
        user_input,
        chat_messages,
        reccurrent_system_prompt,
        intention_analyis_system_prompt,
        mood_analyis_system_prompt,
    ):
        chat_messages.append({"role": "user", "content": user_input})

        messages = []

        messages.append(
            {
                "role": "system",
                "content": reccurrent_system_prompt
                + "\n\n\n"
                + "Also, this chat history is given: \n\n",
            },
        )
        messages.extend(
            chat_messages,
        )
        messages.append(
            {
                "role": "system",
                "content": self._roleplay_system_prompt,
            }
        )

        logging.info(f"messages: {messages}")

        response_message = ""

        for completion in self.client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.5, stream=True
        ):
            response = completion.choices[0].delta.content or ""
            response_message += response
            yield response

        chat_messages.append({"role": "assistant", "content": response_message})

        logging.info(f"response_message: {response_message}")

        mood_analysis_response = mood_analysis(
            self.client, self.name, chat_messages, mood_analyis_system_prompt
        )

        self.current_feeling = mood_analysis_response

        chat_messages.append(
            {
                "role": "system",
                "content": f"{self.name} is feeling: " + self.current_feeling,
            }
        )

        logging.info(f"mood_analysis_response: {mood_analysis_response}")

        intention_analysis_response = intention_analysis(
            self.client, self.name, chat_messages, intention_analyis_system_prompt
        )

        logging.info(f"intention_analysis_response: {intention_analysis_response}")

        self.intention = intention_analysis_response

    def interaction_generator(self, user_input):
        generator = self.response_generator(
            user_input,
            self.chat_messages,
            self._reccurrent_system_prompt,
            self._intention_analyis_system_prompt,
            self._mood_analyis_system_prompt,
        )

        return generator
