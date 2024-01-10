from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Iterator, Union

import logging

from langchain.schema import Document

from utils.constants import FileType, Language

from yt_dlp.utils import DownloadError

import streamlit as st

from langchain.document_loaders.generic import GenericLoader

from openai.error import AuthenticationError, APIConnectionError


from speech_tools.audio_processing import format_time, AudioLoader,\
    SpeechRecognitionParser, CustomYoutubeAudioLoader, WhisperParser

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
    from langchain.document_loaders.blob_loaders import BlobLoader


def get_generator(_loader: BlobLoader,
                   language: Language = Optional[Language.US_English],
                     api_key: Optional[str] = "free") -> Iterator[Document]:
    """
    Returns a generator yielding documents from a BlobLoader and integrates with SpeechRecognition Parser
    """

    # Decided not to use WhisperParser because of high cost and less accurate results for different languages. But is still supported
    # parser = SpeechRecognitionParser(
    #   language=language) if api_key == 'free' else WhisperParser(api_key=api_key, save_dir='audio-chunks/', language=language)

    parser = SpeechRecognitionParser(
        language=language)
    loader = GenericLoader(_loader, parser)
    text_generator = loader.lazy_load()
    return text_generator


class Transcriber:
    '''
        Transcribes speech to text , either using the GoogleSpeechRecognitionAPI or Whisper API
    '''

    def __init__(self, api_key: Optional[str] = 'free'):
        self.type = type
        self.got_input = False
        self.processing = False
        self.docs = []
        self.data = None
        self.api_key = api_key

    def set_container(self, container: DeltaGenerator) -> None:
        """
        sets the streamlit container to display transcribed text in
        """
        self.container = container

    def get_docs(self) -> list:
        return self.docs

    def get_text(self) -> str:
        """
        returns the full transcribed text over all documents
        """
        return '\n'.join([x.page_content for x in self.docs])

    def transcribe(self,
                   data: Union[bytes, str],
                   file_path: str,
                   input_type: FileType,
                   language: Language = Optional[Language.US_English]) -> List[Document]:
        '''
        Displays the Transcribed text using GoogleSpeechRecognitionAPI , results may be inaccurate.

        Args:
            data: The audio data in bytes or youtube url
            file_path: The file path to save the audio at
            input_type: Whether the audio is from a file or from the microphone or a youtube url. Deafaults to File Input.
            language: The language the transcribed text should be in. Defaults to US English.
        Returns:
            docs: List of transcribed documents as langchain Documents
        '''

        logging.debug(f"Transcribe free method is called with File Path:{file_path} , Input Type:{input_type} and\
                      Language:{str(language.value)}")

        # If same audio data is passed , just return already computed documents
        if data == self.data:
            self.container.markdown(f':green[**{self.get_text()}**]')
            return self.docs
        else:
            self.data = data

        self.got_input = False
        self.processing = False

        self.loading_text = self.container.empty()

        if input_type != FileType.YOUTUBE:
            with open(file_path, 'wb') as f:
                f.write(data)

            loader = AudioLoader(file_paths=[file_path])

        else:
            loader = CustomYoutubeAudioLoader([data], save_dir=file_path)

        try:
            with self.loading_text.container():
                st.markdown(
                    f':blue[Speech Processing In Progress...Please Wait...]')

            self.got_input = True
            self.processing = True

            self.container.markdown(f':blue[Transcribed Text:]')
            text_generator = get_generator(loader, language, self.api_key)

            self.docs = []
            for result in text_generator:
                self.loading_text.empty()

                chunk_time = format_time(
                    result.metadata["start_time"]) + ' to ' + format_time(result.metadata["end_time"])

                chunk = result.metadata["chunk"]

                total_chunks = result.metadata["total_chunks"]

                with self.loading_text.container():
                    st.markdown(
                        f':orange[Processed chunk {chunk} / {total_chunks} ({chunk_time})]')

                with self.container:
                    text = result.page_content
                    if text:
                        self.container.markdown(f':green[**{text}**]')
                        logging.debug(
                            f'Transcribed Text of {chunk} : {text}')
                        self.docs.append(result)
                    else:
                        logging.debug(
                            f'Could not transcribe Text of {chunk}')
                        self.container.markdown(
                            f':red[**Could not transcribe audio from {chunk_time}**]')

            return self.docs

        except ValueError as e:
            logging.exception(e)
            self.container.markdown(f':red[{e}]')
        except ConnectionError as e:
            logging.exception(e)
            self.container.markdown(f':red[{e}]')
        except DownloadError as e:
            logging.exception(e)
            self.container.markdown(f':red[Invalid Youtube URL]')
        except AuthenticationError as e:
            logging.exception(e)
            self.container.markdown(f':red[Invalid API Key]')
        except APIConnectionError as e:
            logging.exception(e)
            self.container.markdown(f':red[Error communicating with OpenAI]')
        except Exception as e:
            logging.exception(e)
            self.container.markdown(f':red[{e}]')
        finally:
            self.loading_text.empty()
            self.processing = False
