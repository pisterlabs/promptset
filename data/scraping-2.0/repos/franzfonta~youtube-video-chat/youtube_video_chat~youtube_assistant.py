import logging
import time
from typing import Callable, Optional

from openai import OpenAI

from youtube_video_chat.youtube_thread import YouTubeThread


class YouTubeAssistant:
    """
    Represents an assistant for interacting with YouTube videos and transcripts.
    """

    def __init__(self, client: OpenAI, assistant_id: str, transcript_fetcher: Callable[[str], str]):
        """
        Initializes a new instance of the YouTubeAssistant class by retrieving the existing
        assistant.

        Args:
            client: The client object used to interact with the YouTube API.
            transcript_fetcher: An instance of the YouTubeTranscriptFetcher class used to
            fetch video transcripts.
        """
        self.client = client
        self.transcript_fetcher = transcript_fetcher
        self.assistant = client.beta.assistants.retrieve(assistant_id)

    def create_thread(self, video_url: str) -> YouTubeThread:
        """
        Creates the thread for a YouTube video and sends the first message with the transcript.

        Args:
            video_url (str): The URL of the YouTube video.

        Returns:
            YouTubeThread: The created YouTubeThread object.
        """

        transcript = self.transcript_fetcher(video_url)

        openai_thread = self.client.beta.threads.create()

        # TODO persist thread ID for later retrieval

        youtube_thread = YouTubeThread(video_url, transcript, openai_thread)

        # Create the first message in the thread with the video transcript
        initial_prompt = f"This is the transcript of a YouTube video: \
            \n\"{transcript}\".\n \
            In the following messages I will ask you questions about it. \
            As for now, summarize the video in 100 words or less."
        self.ask_question(youtube_thread, initial_prompt, True)

        return youtube_thread

    def __retrieve_run(self, thread_id: str, run_id: str, max_retries: int = 5, base_delay: int = 2):
        """
        Retrieve a run from a thread until it is completed or maximum retries are reached.

        Args:
            thread_id (str): The ID of the thread.
            run_id (str): The ID of the run.
            max_retries (int, optional): The maximum number of retries. Defaults to 5.
            base_delay (int, optional): The base delay in seconds. Defaults to 2.

        Returns:
            The completed run.

        Raises:
            Exception: If maximum retries are reached and the operation fails.
        """
        # Poll the run until it is completed
        retries = 0
        while retries < max_retries:
            logging.info(f"Attempt {retries + 1}")
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run_id)
            if run.status == "completed":
                return run
            else:
                retries += 1
                delay = base_delay * 2 ** retries
                logging.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        raise Exception("Max retries reached, operation failed.")

    def ask_question(self, thread: YouTubeThread, prompt: str, is_initial_prompt: bool = False) -> Optional[str]:
        """
        Sends a question to the YouTube Assistant and retrieves the response.

        Args:
            thread (YouTubeThread): The YouTube thread to send the question to.
            prompt (str): The question prompt.
            is_initial_prompt (bool, optional): True if the prompt is the initial prompt. Defaults to False.

        Returns:
            Optional[str]: The response from the YouTube Assistant or None if the operation fails.
        """
        # Add user message to thread except for the initial prompt
        if not is_initial_prompt:
            thread.messages.append({"role": "user", "content": prompt})

        try:
            # Create a new message in the thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread.openai_thread.id,
                role="user",
                content=prompt
            )

            # Create a new run
            run = self.client.beta.threads.runs.create(
                thread_id=thread.openai_thread.id,
                assistant_id=self.assistant.id
            )

            # Wait for the run to complete
            run = self.__retrieve_run(thread.openai_thread.id, run.id)

            # Retrieve the last message in the thread
            messages = self.client.beta.threads.messages.list(
                thread_id=thread.openai_thread.id)
            response = messages.data[0].content[0].text.value

            # Add assistant response to chat history
            thread.messages.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            logging.error(e)
            return None
