from unittest import mock

import django
import pytest
from aiohttp import ClientResponseError
from openai.error import ServiceUnavailableError, RateLimitError

import bobweb.bob.config
from bobweb.bob.command import ChatCommand
from bobweb.bob.command_speech import SpeechCommand
from bobweb.bob.tests_mocks_v2 import init_chat_user
from bobweb.bob.tests_utils import assert_command_triggers


async def speech_api_mock_response_200(*args, **kwargs):
    return str.encode('this is hello.mp3 in bytes')

async def speech_api_mock_response_client_response_error(*args, **kwargs):
    raise ClientResponseError(status=-1, message='mock error message', request_info=None, history=None)

async def speech_api_mock_response_service_unavailable_error(*args, **kwargs):
    raise ServiceUnavailableError()

async def speech_api_mock_response_rate_limit_error_error(*args, **kwargs):
    raise RateLimitError()

@pytest.mark.asyncio
@mock.patch('bobweb.bob.openai_api_utils.user_has_permission_to_use_openai_api', lambda *args: True)
@mock.patch('bobweb.bob.async_http.post_expect_bytes', speech_api_mock_response_200)
class SpeechCommandTest(django.test.TransactionTestCase):
    bobweb.bob.config.openai_api_key = 'DUMMY_VALUE_FOR_ENVIRONMENT_VARIABLE'
    command_class: ChatCommand.__class__ = SpeechCommand
    command_str: str = 'lausu'

    @classmethod
    def setUpClass(cls) -> None:
        super(SpeechCommandTest, cls).setUpClass()
        cls.maxDiff = None

    async def test_command_triggers(self):
        should_trigger = [f'/{self.command_str}', f'!{self.command_str}', f'.{self.command_str}',
                          f'/{self.command_str.upper()}', f'/{self.command_str.upper()} test']
        should_not_trigger = [f'{self.command_str}', f'test /{self.command_str}']
        await assert_command_triggers(self, self.command_class, should_trigger, should_not_trigger)

    async def test_when_no_parameter_and_not_reply_gives_help_text(self):
        chat, user = init_chat_user()
        await user.send_message('/lausu')
        self.assertEqual('Kirjoita lausuttava viesti komennon \'\\lausu\' jälkeen ' \
                         'tai lausu toinen viesti vastaamalla siihen pelkällä komennolla',
                         chat.last_bot_txt())

    async def test_when_no_parameter_and_reply_with_no_to_text_gives_help_text(self):
        chat, user = init_chat_user()
        message = await user.send_message('')
        await user.send_message('/lausu', reply_to_message=message)
        self.assertEqual('Kirjoita lausuttava viesti komennon \'\\lausu\' jälkeen ' \
                         'tai lausu toinen viesti vastaamalla siihen pelkällä komennolla',
                         chat.last_bot_txt())

    async def test_when_ok_parameter_but_also_reply_gives_parameter_as_speech(self):
        chat, user = init_chat_user()
        message = await user.send_message('should not translate')
        await user.send_message('/lausu hello', reply_to_message=message)
        self.assertEqual('hello',
                         chat.last_bot_txt())

    async def test_too_long_title_gets_cut(self):
        chat, user = init_chat_user()
        message = await user.send_message('this is a too long prompt to be in title fully')
        await user.send_message('/lausu', reply_to_message=message)
        self.assertEqual('this is a ',
                         chat.last_bot_txt())

    async def test_client_response_error(self):
        chat, user = init_chat_user()
        message = await user.send_message('hello')
        with (
            self.assertLogs(level='ERROR') as log,
            mock.patch(
                'bobweb.bob.async_http.post_expect_bytes',
                speech_api_mock_response_client_response_error)):
            await user.send_message('/lausu', reply_to_message=message)
            self.assertIn('Openai /v1/audio/speech request returned with ' \
                          'status: -1. Response text: \'mock error message\'',
                          log.output[-1])
            self.assertEqual(
                'OpenAI:n api vastasi pyyntöön statuksella -1',
                chat.last_bot_txt())

    async def test_service_unavailable_error(self):
        chat, user = init_chat_user()
        message = await user.send_message('hello')
        with (
            mock.patch(
                'bobweb.bob.async_http.post_expect_bytes',
                speech_api_mock_response_service_unavailable_error)):
            await user.send_message('/lausu', reply_to_message=message)
            self.assertEqual(
                'OpenAi:n palvelu ei ole käytettävissä ' \
                'tai se on juuri nyt ruuhkautunut. ' \
                'Ole hyvä ja yritä hetken päästä uudelleen.',
                chat.last_bot_txt())

    async def test_rate_limit_error(self):
        chat, user = init_chat_user()
        message = await user.send_message('hello')
        with (
            mock.patch(
                'bobweb.bob.async_http.post_expect_bytes',
                speech_api_mock_response_rate_limit_error_error)):
            await user.send_message('/lausu', reply_to_message=message)
            self.assertEqual(
                'OpenAi:n palvelu ei ole käytettävissä ' \
                'tai se on juuri nyt ruuhkautunut. ' \
                'Ole hyvä ja yritä hetken päästä uudelleen.',
                chat.last_bot_txt())
