import io
from unittest import mock

import openai
import pytest
import django
from django.test import TestCase
from pydub.exceptions import CouldntDecodeError

from telegram import Voice, File

from bobweb.bob import main, database, message_handler_voice
from bobweb.bob.message_handler_voice import TranscribingError
from bobweb.bob.tests_mocks_v2 import init_chat_user, MockChat


async def openai_api_mock_response_with_transcription(*args, **kwargs):
    return {"text": "this is mock transcription"}


def create_mock_converter(written_bytes: int):
    """ Returns mock function that returns empty Bytes object and given
        number as written_bytes buffer size """

    def mock_implementation(*args):
        return io.BytesIO(), written_bytes

    return mock_implementation


def create_mock_converter_that_raises_exception(exception: Exception):
    """ Returns mock function that raises exception given as parameter """

    def mock_implementation(*args):
        raise exception

    return mock_implementation


class MockVoice(Voice):
    default_file_id = 'AwACAgQAAxkBAAIQS2RFXO0thVNH86FUcCwpNK7aHDjUAAJKDgAC7AUgUvVxjAac8EeILwQ'
    default_unique_id = 'AgADSg4AAuwFIFI'

    def __init__(self, audio_file: io.BytesIO | bytes, file_id: str = default_unique_id, file_unique_id: str = default_file_id, duration: int = 1):
        super().__init__(file_id, file_unique_id, duration)
        super()._unfreeze()
        self.file_size = 30217
        self.mime_type = 'audio/ogg'
        self.file: File = MockFile(audio_file)

    async def get_file(self, *args) -> "File":
        return self.file


class MockFile(File):
    default_file_id = 'AwACAgQAAxkBAAIQS2RFXO0thVNH86FUcCwpNK7aHDjUAAJKDgAC7AUgUvVxjAac8EeILwQ'
    default_file_unique_id = 'AgADSg4AAuwFIFI'

    def __init__(self, audio_file: io.BytesIO | bytes, file_id: str = default_file_id, file_unique_id: str = default_file_unique_id):
        super().__init__(file_id, file_unique_id)
        super()._unfreeze()
        self.file_path = 'https://api.telegram.org/file/bot5057789773:AAGWzH5YYEaSwqDyaJ-Bqg3GgtJ7d1yVVV0/voice/file_1.oga'
        self.file_size = 30217
        # Set audio file that is returned by download_call
        self.audio_file = audio_file

    async def download_to_memory(self, out, *args) -> None:
        out.write(self.audio_file)


def create_mock_voice(audio_file: io.BytesIO | bytes = None) -> Voice:
    audio_file = audio_file or io.BytesIO().read()
    return MockVoice(audio_file)


async def create_chat_and_user_and_try_to_transcribe_audio() -> MockChat:
    """ Common test pattern extracted to method """
    chat, user = init_chat_user()
    voice: Voice = create_mock_voice()
    voice_msg = await user.send_voice(voice)
    await user.send_message('/tekstitä', reply_to_message=voice_msg)
    return chat


@pytest.mark.asyncio
@mock.patch('bobweb.bob.async_http.post_expect_json', openai_api_mock_response_with_transcription)
@mock.patch('bobweb.bob.openai_api_utils.user_has_permission_to_use_openai_api', lambda *args: True)
class VoiceMessageHandlerTest(django.test.TransactionTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(VoiceMessageHandlerTest, cls).setUpClass()
        cls.maxDiff = None
        openai.api_key = 'api_key_value'

    async def test_that_ffmpeg_is_available_in_running_environment(self):
        fail_msg = 'ffmpeg program not available as runnable console command in the running environment. ' \
                   'Install ffmpeg to enable bot\'s features that are dependant on it. For more info ' \
                   'check https://ffmpeg.org/'
        self.assertTrue(message_handler_voice.ffmpeg_available, fail_msg)

    async def test_voice_message_should_be_automatically_transcribed_when_settings_are_accordingly(self):
        """ Uses ffmpeg to convert a real ogg file to mp4. Tests that the voice message is automatically
            transcribed when the chat has 'voice_to_text_enabled' == True """
        with open('bobweb/bob/resources/test/telegram_voice_message_mock.ogg', "rb") as test_sound_file:
            chat, user = init_chat_user()
            chat_entity = database.get_chat(chat.id)
            chat_entity.voice_msg_to_text_enabled = True
            chat_entity.save()

            voice: Voice = create_mock_voice(test_sound_file.read())
            await user.send_voice(voice)

            self.assertIn('"this is mock transcription"', chat.last_bot_txt())
            self.assertIn('Rahaa paloi: $0.000100, rahaa palanut rebootin jälkeen: $0.000100', chat.last_bot_txt())

    @mock.patch('bobweb.bob.message_handler_voice.convert_buffer_content_to_audio',
                create_mock_converter_that_raises_exception(TranscribingError('[Reason]')))
    async def test_gives_error_message_if_transcribing_error_is_raised(self):
        chat = await create_chat_and_user_and_try_to_transcribe_audio()
        self.assertIn('Median tekstittäminen ei onnistunut. [Reason]', chat.last_bot_txt())

    @mock.patch('bobweb.bob.message_handler_voice.convert_buffer_content_to_audio',
                create_mock_converter(1024 ** 2 * 25 + 1))
    async def test_gives_error_if_voice_file_over_25_MB(self):
        # As the buffer size 1 byte over 25 MB, should return error that states the file is too big
        chat = await create_chat_and_user_and_try_to_transcribe_audio()
        self.assertIn('Äänitiedoston koko oli liian suuri.', chat.last_bot_txt())

    @mock.patch('bobweb.bob.message_handler_voice.convert_buffer_content_to_audio',
                create_mock_converter_that_raises_exception(CouldntDecodeError()))
    async def test_gives_error_message_decoding_error_is_raised(self):
        chat = await create_chat_and_user_and_try_to_transcribe_audio()
        expected_msg = 'Ääni-/videotiedoston alkuperäistä tiedostotyyppiä tai sen sisältämää median koodekkia ei tueta,'
        self.assertIn(expected_msg, chat.last_bot_txt())

    @mock.patch('bobweb.bob.message_handler_voice.convert_buffer_content_to_audio',
                create_mock_converter_that_raises_exception(Exception()))
    async def test_catches_any_exception_and_gives_error_msg(self):
        chat = await create_chat_and_user_and_try_to_transcribe_audio()
        expected_msg = 'Median tekstittäminen ei onnistunut odottamattoman poikkeuksen johdosta.'
        self.assertIn(expected_msg, chat.last_bot_txt())
