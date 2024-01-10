import os
import typing

import openai
from openai.openai_object import OpenAIObject

from ..database.stories import Page, Story
from ..web.exceptions import SuessException


class ChatCompletionMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def get_message(self):
        return {"role": self.role, "content": self.content}


SYSTEM_MESSAGE = ChatCompletionMessage("system", "You're story teller for kids.")

INTRO_MESSAGE_TEMPLATE = """
Together we will build a story. Here are the rules:
- The story is 5-10 passages long
- Each passage will be around 150 words
- At the end of each passage you prompt me with an open-ended question about a decision that I need to make.
- The story should be written for me, a 10 year old 3rd grade student. Make sure the story is appropriate for a user of my age.
- Treat me as the protagonist of the story."""

THEME_RULE_TEMPLATE = "\n- The story will include the themes: {themes}"
TOPIC_RULE_TEMPLATE = "\n- The story will include the topics: {topics}"
CONTEXT_RULE_TEMPLATE = "\n- Heres some additional context on the story: {context}"

ADDITIONAL_CONTEXT = ChatCompletionMessage(
    "user",
    """Additionally, here's some more additional context to incoporate into my story.
    My name is Jared and I'm wearing a red hat.
    """,
)


class GPTStoryManager:
    story: Story

    def __init__(self, story: Story):
        self.story = story
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SuessException(500, "OPENAI_API_KEY not set")
        else:
            openai.api_key = api_key

    def get_next_page(self, input: str) -> str:
        """Given the story, this method will return the output for the given input.
        The output represents the newly created "page", in reponse to the given input.

        Args:
            input (str): The user input for the decision made based on the previous page

        Raises:
            SuessException if something went wrong retrieving the output from ChatGPT.

        Returns:
            str: the output from chat gpt
        """
        messages = [self._get_story_intro_messages()] + self._get_story_pages_messages()
        messages.append(ChatCompletionMessage("user", input))
        response = self._get_gpt_response(messages)
        return response

    def _get_story_intro_messages(self) -> ChatCompletionMessage:
        """Get the introduction messages required for ChatCompletionRequest

        Returns:
            typing.List[ChatCompletionMessage]: introduction messages including the story's initial context and the system prompt
        """
        themes = [theme.theme for theme in self.story.themes]
        topics = [topic.topic for topic in self.story.topics]
        context = self.story.additional_context
        intro_message_content = f"{INTRO_MESSAGE_TEMPLATE}{THEME_RULE_TEMPLATE.format(themes=', '.join(themes)) if themes else ''}{TOPIC_RULE_TEMPLATE.format(topics=', '.join(topics)) if topics else ''}{CONTEXT_RULE_TEMPLATE.format(context=context) if context else ''}"
        return ChatCompletionMessage("user", intro_message_content)

    def _get_story_pages_messages(self) -> typing.List[ChatCompletionMessage]:
        """Gets the list of chatGPT completion messages for each page in the story

        Returns:
            typing.List[ChatCompletionMessage]: list of ChatCompletionMessage for each page in the story
        """
        messages = []
        for page in self.story.pages:
            page: Page
            messages.append(ChatCompletionMessage("user", page.input))
            messages.append(ChatCompletionMessage("assistant", page.output))
        return messages

    def _get_gpt_response(self, messages: typing.List[ChatCompletionMessage]) -> str:
        """Get the string response of the first choice in a gpt chat completion.

        Args:
            messages (typing.List[ChatCompletionMessage]): context + new message prompt

        Raises:
            SuessException if response failed, or parsing failed for any reason

        Returns:
            str: the response from GPT api
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=list(map(lambda msg: msg.get_message(), messages)),
        )
        return GPTStoryManager._handle_response(response)

    @staticmethod
    def _handle_response(response: OpenAIObject) -> str:
        """Given a chatGPT api request, returns the string response

        Args:
            response (OpenAIObject): response from ChatGPT api request

        Raises:
            SuessException: if parsing fails for any reason. Some reasons include, malformed dict, finish_reason isn't stop, content not found

        Returns:
            str: the content of the first choice from the chatgpt api response
        """
        choices = response.get("choices")
        if choices:
            response: OpenAIObject = choices[0]
            content = response.get("message", {}).get("content")
            if response.get("finish_reason") == "stop" and content:
                return content
        raise SuessException(
            500, f"Something went wrong with OpenAI response. Response: {response}"
        )
