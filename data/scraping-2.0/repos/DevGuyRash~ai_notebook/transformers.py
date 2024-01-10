import openai
import filetypes
import os
from environment import Environment


class Transcript:
    """
    Class to transcribe any YouTube video to text.

    Attributes:
        _environment (Environment): `Environment` object for getting the
            api key and various other environment variables.
        video (filetypes.YtVideo): `filetypes.YtVideo` object containing
            the video to get the transcripts for.
        audio_filepath (str): Filepath of the audio file.
    """

    def __init__(self,
                 output_filepath: str = "transcript.txt",
                 audio_filepath: str = ""
                 ):
        # TODO: Pass api key to environment object
        """
        Initializes required variables for the `Transcript` object.

        Args:
            output_filepath: Filepath of where the transcript will be
                saved.
            audio_filepath: Filepath of where the audio file will be
                saved.
        """
        self._environment = Environment()

        self.video = None
        self.audio_filepath = audio_filepath

        # Init transcription variables
        self._transcript = None
        self._transcript_filepath = output_filepath

    def _create_video(self,
                      video_url: str = "",
                      *args,
                      **kwargs,
                      ):
        """
        Sets the `filetypes.YtVideo` object with a valid YouTube url.

        The user is prompted for a valid YouTube video url. Using the
        valid url, a new `filetypes.YtVideo` object is created and bound
         to `video`.

        Args:
            *args: Additional arguments to send to the
                `filetypes.YtVideo` object.
            video_url: Valid YouTube video url. Optional.
            **kwargs: Additional keyword arguments to send to the
                `filetypes.YtVideo` object.
        """
        print(f"Fetching video for: {video_url}")
        if not video_url:
            video_url = input("Enter a youtube video url: ")

        self.video = filetypes.YtVideo(video_url, *args, **kwargs)

    def transcribe_yt_video(self,
                            video_url: str = "",
                            remove_audio: bool = True,
                            *args,
                            **kwargs,
                            ) -> None:
        """
        Transcribes a YouTube video to text using OpenAIs Whisper

        Accepts a YouTube url in `video_url` if one does not already
        exist in the object. The video is passed to OpenAIs whisper
        and transcribed. The text is then saved to a file specified by
        `transcript_filepath`.

        Args:
            video_url: YouTube video to transcribe.
            remove_audio: Whether to delete the audio file after it's
                done being used for transcription.
            args: Any additional arguments to be passed to a
                `filetypes.YtVideo` object.
            kwargs: Any additional arguments to be passed to a
                `filetypes.YtVideo` object.
        """
        # Check that video exists or user provided an url
        if not self.video or video_url:
            self._create_video(*args, video_url=video_url, **kwargs)

        # Save audio before attempting to transcribe
        self.video.save_audio_file()
        self.set_audio_filepath(self.video.audio_filepath)

        # Transcribe audio
        self.transcribe_audio_file(self.get_audio_filepath())

        if remove_audio:
            self.delete_file(self.audio_filepath)

    def transcribe_audio_file(self,
                              audio_filepath: str = "",
                              ) -> None:
        """Gets valid audio filepath for `_transcribe_audio_file`"""
        if not self.audio_filepath and not audio_filepath:
            self.set_audio_filepath(input("Enter the filepath (or filename if "
                                          "it's in the same directory) of the "
                                          "audio file:\n>"))
        elif audio_filepath:
            # self.audio_filepath doesn't exist but audio_filepath arg does
            self.set_audio_filepath(audio_filepath)

        # Verify that audio file exists
        while not os.path.exists(self.audio_filepath):
            print(f"Invalid audio filepath: {self.audio_filepath}")
            self.set_audio_filepath(input("Please enter a new one:\n>"))

        # Transcribe audio
        self._transcribe_audio_file(self.audio_filepath)

    def _transcribe_audio_file(self,
                               audio_filepath: str,
                               ) -> None:
        """Transcribes audio file using OpenAIs whisper."""
        audio_file = open(audio_filepath, 'rb')
        try:
            print("Attempting to transcribe audio...")
            # Get transcript
            self._transcript = openai.Audio.transcribe(
                "whisper-1",
                file=audio_file,
            ).get("text")
            print("Successfully transcribed audio.")
        except openai.error.AuthenticationError as error:
            # No API Key was set, or an incorrect one provided.
            print("===========ERROR==============")
            print(f"Error: {error.error}")
            print(f"Code: {error.code}")
            print(f"Message: {error.user_message}")
            print("==============================")
        else:
            self.save_transcript()
        finally:
            audio_file.close()

    def save_transcript(self) -> None:
        """Saves `transcript` to a text file at `transcript_filepath`"""
        with open(self._transcript_filepath, "w", encoding='utf-8') \
                as file:
            # Write transcript to file
            print("Saving transcript...")
            print(self._transcript, file=file)
            print(f"Saved transcript to: {self._transcript_filepath}")

    def print_transcript(self) -> None:
        """Prints transcript text."""
        print("Transcript:")
        print(self._transcript)

    def set_api_key(self, api_key: str = "") -> None:
        """Sets the api key for OpenAI"""
        self._environment.set_openai_api_key(api_key)

    def get_api_key(self) -> str:
        """Returns current API key if one exists"""
        return self._environment.get_api_key()

    def set_audio_filepath(self, filepath: str) -> None:
        """Sets the audio filepath."""
        self.audio_filepath = filepath

    def get_audio_filepath(self) -> str:
        """Returns the audio filepath"""
        return self.audio_filepath

    @staticmethod
    def delete_file(filepath: str):
        """Deletes `filepath` from system."""
        path_head, path_tail = os.path.split(filepath)
        # Display path_head if path_tail is empty due to trailing slash
        print(f"Removing file: {path_tail or path_head}")
        os.remove(filepath)
        print(f"Audio file removed at: {filepath}")
