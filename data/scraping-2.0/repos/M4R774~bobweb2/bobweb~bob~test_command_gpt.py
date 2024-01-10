import io
import os

import pytest
from django.core import management
from django.test import TestCase
from unittest import mock

from telegram import PhotoSize

import bobweb
from bobweb.bob import main, database, command_gpt, openai_api_utils
from bobweb.bob.openai_api_utils import remove_cost_so_far_notification_and_context_info, ResponseGenerationException
from bobweb.bob.test_command_speech import speech_api_mock_response_service_unavailable_error, \
    speech_api_mock_response_rate_limit_error_error
from bobweb.bob.tests_mocks_v2 import MockTelethonClientWrapper, init_chat_user

from bobweb.bob.command_gpt import GptCommand, generate_no_parameters_given_notification_msg, \
    remove_gpt_command_related_text, determine_used_model

import django

from bobweb.bob.tests_utils import assert_command_triggers, assert_get_parameters_returns_expected_value, AsyncMock, \
    get_json
from bobweb.web.bobapp.models import Chat

os.environ.setdefault(
    'DJANGO_SETTINGS_MODULE',
    'bobweb.web.web.settings'
)

os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = 'true'
django.setup()


class MockOpenAIObject:
    def __init__(self):
        self.choices = [Choice()]
        self.usage = Usage()


class Choice:
    def __init__(self):
        self.message = Message()


class Message:
    def __init__(self):
        # https://platform.openai.com/tokenizer: 53 characters, 13 tokens.
        self.content = 'The Los Angeles Dodgers won the World Series in 2020.'
        self.role = 'assistant'


class Usage:
    def __init__(self):
        self.prompt_tokens = 16
        self.completion_tokens = 26
        self.total_tokens = 42


def assert_gpt_api_called_with(mock_method: AsyncMock, model: str, messages: list[dict[str, str]]):
    """
    Helper method for determining on how OpenAi http API endpoint was called. Added when Gpt was switched
    from openai python library to direct http requests.
    """
    mock_method.assert_called_with(
        url='https://api.openai.com/v1/chat/completions',
        headers={'Authorization': 'Bearer DUMMY_VALUE_FOR_ENVIRONMENT_VARIABLE'},
        json={'model': model, 'messages': messages, 'max_tokens': 4096}
    )


async def mock_response_from_openai(*args, **kwargs):
    return get_json(MockOpenAIObject())


async def raises_response_generation_exception(*args, **kwargs):
    raise ResponseGenerationException('response generation raised an exception')


@mock.patch('bobweb.bob.async_http.post_expect_json', mock_response_from_openai)
@mock.patch('bobweb.bob.openai_api_utils.user_has_permission_to_use_openai_api', lambda *args: True)
@pytest.mark.asyncio
class ChatGptCommandTests(django.test.TransactionTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(ChatGptCommandTests, cls).setUpClass()
        django.setup()
        management.call_command('migrate')
        bobweb.bob.config.openai_api_key = 'DUMMY_VALUE_FOR_ENVIRONMENT_VARIABLE'

    async def test_command_triggers(self):
        should_trigger = ['/gpt', '!gpt', '.gpt', '/GPT', '/gpt test', '/gpt3', '/gpt3.5', '/gpt4']
        should_not_trigger = ['gpt', 'test /gpt', '/gpt2', '/gpt3.0', '/gpt4.0', '/gpt5']
        await assert_command_triggers(self, GptCommand, should_trigger, should_not_trigger)

    async def test_get_given_parameter(self):
        assert_get_parameters_returns_expected_value(self, '!gpt', command_gpt.instance)

    async def test_no_prompt_gives_help_reply(self):
        chat, user = init_chat_user()
        expected_reply = generate_no_parameters_given_notification_msg()
        await user.send_message('/gpt')
        self.assertEqual(expected_reply, chat.last_bot_txt())

    async def test_should_contain_correct_response(self):
        openai_api_utils.state.reset_cost_so_far()
        chat, user = init_chat_user()
        await user.send_message('/gpt Who won the world series in 2020?')
        expected_reply = 'The Los Angeles Dodgers won the World Series in 2020.' \
                         '\n\nKonteksti: 1 viesti. Rahaa paloi: $0.000940, rahaa palanut rebootin jälkeen: $0.000940'
        self.assertEqual(expected_reply, chat.last_bot_txt())

    async def test_set_new_system_prompt(self):
        chat, user = init_chat_user()
        await user.send_message('.gpt .system uusi homma')
        self.assertEqual('System-viesti asetettu annetuksi.', chat.last_bot_txt())

    async def test_each_command_without_replied_messages_is_in_its_own_context(self):
        openai_api_utils.state.reset_cost_so_far()
        chat, user = init_chat_user()
        # 3 commands are sent. Each has context of 1 message and same cost per message, however
        # total cost has accumulated.
        for i in range(1, 4):
            await user.send_message(f'.gpt Konteksti {i}')
            self.assertIn(f"Konteksti: 1 viesti. Rahaa paloi: $0.000940, "
                          f"rahaa palanut rebootin jälkeen: ${get_cost_str(i)}", chat.last_bot_txt())

    async def test_context_content(self):
        """ A little bit more complicated test. Tests that messages in reply threads are included
            in the next replies message context as expected. Here we create first a chain of
            three gpt-command that each are replies to previous commands answer from bot. Each
            bots answer is reply to the command that triggered it. So there is a continuous
            reply-chain from the first gpt-command to the last reply from bot"""
        openai_api_utils.state.reset_cost_so_far()
        chat, user = init_chat_user()
        await user.send_message('.gpt .system uusi homma')
        self.assertEqual('System-viesti asetettu annetuksi.', chat.last_bot_txt())
        prev_msg_reply = None

        # Use mock telethon client wrapper that does not try to use real library but instead a mock
        # that searches mock-objects from initiated chats bot-objects collections
        with mock.patch('bobweb.bob.telethon_service.client', MockTelethonClientWrapper(chat.bot)):
            for i in range(1, 4):
                # Send 3 messages where each message is reply to the previous one
                await user.send_message(f'.gpt Konteksti {i}', reply_to_message=prev_msg_reply)
                prev_msg_reply = chat.last_bot_msg()
                messages_text = 'viesti' if i == 1 else 'viestiä'
                self.assertIn(f"Konteksti: {1 + (i - 1) * 2} {messages_text}. Rahaa paloi: $0.000940, "
                              f"rahaa palanut rebootin jälkeen: ${get_cost_str(i)}", chat.last_bot_txt())

            # Now that we have create a chain of 6 messages (3 commands, and 3 answers), add
            # one more reply to the chain and check, that the MockApi is called with all previous
            # messages in the context (in addition to the system message)
            mock_method = AsyncMock()
            mock_method.return_value = get_json(MockOpenAIObject())
            with mock.patch('bobweb.bob.async_http.post_expect_json', mock_method):
                await user.send_message('/gpt Who won the world series in 2020?', reply_to_message=prev_msg_reply)

            expected_call_args_messages = [
                {'role': 'system', 'content': 'uusi homma', },
                {'role': 'user', 'content': 'Konteksti 1'},
                {'role': 'assistant', 'content': 'The Los Angeles Dodgers won the World Series in 2020.'},
                {'role': 'user', 'content': 'Konteksti 2', },
                {'role': 'assistant', 'content': 'The Los Angeles Dodgers won the World Series in 2020.'},
                {'role': 'user', 'content': 'Konteksti 3', },
                {'role': 'assistant', 'content': 'The Los Angeles Dodgers won the World Series in 2020.'},
                {'role': 'user', 'content': 'Who won the world series in 2020?'}
            ]
            assert_gpt_api_called_with(mock_method, model='gpt-4-1106-preview', messages=expected_call_args_messages)

    async def test_no_system_message(self):
        openai_api_utils.state.reset_cost_so_far()
        chat, user = init_chat_user()
        mock_method = AsyncMock()
        mock_method.return_value = get_json(MockOpenAIObject())

        with (
            mock.patch('bobweb.bob.telethon_service.client', MockTelethonClientWrapper(chat.bot)),
            mock.patch('bobweb.bob.async_http.post_expect_json', mock_method)
        ):
            await user.send_message('.gpt test')
            expected_call_args_messages = [{'role': 'user', 'content': 'test'}]
            assert_gpt_api_called_with(mock_method, model='gpt-4-1106-preview', messages=expected_call_args_messages)

            # Now, if system message is added, it is included in call after that
            await user.send_message('.gpt .system system message')
            await user.send_message('.gpt test2')
            expected_call_args_messages = [
                {'role': 'system', 'content': 'system message'},
                {'role': 'user', 'content': 'test2'}
            ]
            assert_gpt_api_called_with(mock_method, model='gpt-4-1106-preview', messages=expected_call_args_messages)

    async def test_gpt_command_without_any_message_as_reply_to_another_message(self):
        """
        Tests that if user replies to another message with just '/gpt' command, then that
        other message (and any messages in the reply chain) are included in the api calls
        context message history. The '/gpt' command message itself is not included, as it
        contains nothing else than the command itself.
        """
        openai_api_utils.state.reset_cost_so_far()
        chat, user = init_chat_user()
        mock_method = AsyncMock()
        mock_method.return_value = get_json(MockOpenAIObject())

        with (
            mock.patch('bobweb.bob.telethon_service.client', MockTelethonClientWrapper(chat.bot)),
            mock.patch('bobweb.bob.async_http.post_expect_json', mock_method)
        ):
            original_message = await user.send_message('some message')
            gpt_command_message = await user.send_message('.gpt', reply_to_message=original_message)
            expected_call_args_messages = [{'role': 'user', 'content': 'some message'}]
            assert_gpt_api_called_with(mock_method, model='gpt-4-1106-preview', messages=expected_call_args_messages)

            # Now, if there is just a gpt-command in the reply chain, that message is excluded from
            # the context message history for later calls
            await user.send_message('/gpt something else', reply_to_message=gpt_command_message)
            expected_call_args_messages = [{'role': 'user', 'content': 'some message'},
                                           {'role': 'user', 'content': 'something else'}]
            assert_gpt_api_called_with(mock_method, model='gpt-4-1106-preview', messages=expected_call_args_messages)

    async def test_prints_system_prompt_if_sub_command_given_without_parameters(self):
        # Create a new chat. Expect bot to tell, that system msg is empty
        chat, user = init_chat_user()
        await user.send_message('/gpt /system')
        self.assertIn('Nykyinen system-viesti on nyt tyhjä', chat.last_bot_txt())

        # Now set system message for chat and check that it is contained in the response
        chat_entity = Chat.objects.get(id=chat.id)
        chat_entity.gpt_system_prompt = '_system_prompt_'
        chat_entity.save()

        await user.send_message('/gpt /system')
        self.assertIn('Nykyinen system-viesti on nyt:\n\n_system_prompt_', chat.last_bot_txt())

    async def test_system_prompt_can_be_updated_with_sub_command(self):
        # Create a new chat. Expect bot to tell, that system msg is empty
        chat, user = init_chat_user()
        await user.send_message('/gpt /system')
        self.assertIn('Nykyinen system-viesti on nyt tyhjä', chat.last_bot_txt())

        # Give command to update system message, check from database that it has been updated
        await user.send_message('/gpt /system _new_system_prompt_')

        chat_entity = Chat.objects.get(id=chat.id)
        self.assertEqual('_new_system_prompt_', chat_entity.gpt_system_prompt)

    async def test_system_prompt_is_chat_specific(self):
        # Initiate 2 different chats that have cc-holder as member
        # cc-holder user is ignored as it's not needed in this test case
        chat_a, user_a = init_chat_user()
        b_chat, b_user = init_chat_user()

        # Both users send message with gpt command to their corresponding chats
        await user_a.send_message('/gpt /system')
        await b_user.send_message('/gpt /system')

        self.assertIn('Nykyinen system-viesti on nyt tyhjä', chat_a.last_bot_txt())
        self.assertIn('Nykyinen system-viesti on nyt tyhjä', b_chat.last_bot_txt())

        # Update chat a system message and check that it has changed in the database, but chat b has not
        await user_a.send_message('/gpt /system AAA')
        self.assertEqual('AAA', Chat.objects.get(id=chat_a.id).gpt_system_prompt)
        self.assertIsNone(Chat.objects.get(id=b_chat.id).gpt_system_prompt)

        # Update chat b system message and check that it has changed in the database
        await b_user.send_message('/gpt /system B')
        self.assertEqual('AAA', Chat.objects.get(id=chat_a.id).gpt_system_prompt)
        self.assertEqual('B', Chat.objects.get(id=b_chat.id).gpt_system_prompt)

    async def test_quick_system_prompt(self):
        mock_method = AsyncMock()
        mock_method.return_value = get_json(MockOpenAIObject())
        with mock.patch('bobweb.bob.async_http.post_expect_json', mock_method):
            chat, user = init_chat_user()
            await user.send_message('hi')  # Saves user and chat to the database
            chat_entity = Chat.objects.get(id=chat.id)
            chat_entity.quick_system_prompts = {'1': 'this is a test quick system message'}
            chat_entity.save()
            await user.send_message('/gpt /1 Who won the world series in 2020?')

            expected_call_args = [{'role': 'system', 'content': 'this is a test quick system message'},
                                  {'role': 'user', 'content': 'Who won the world series in 2020?'}]
            assert_gpt_api_called_with(mock_method, model='gpt-4-1106-preview', messages=expected_call_args)

    async def test_another_quick_system_prompt(self):
        mock_method = AsyncMock()
        mock_method.return_value = get_json(MockOpenAIObject())
        with mock.patch('bobweb.bob.async_http.post_expect_json', mock_method):
            chat, user = init_chat_user()
            await user.send_message('hi')  # Saves user and chat to the database
            chat_entity = Chat.objects.get(id=chat.id)
            chat_entity.quick_system_prompts = {'2': 'this is a test quick system message'}
            chat_entity.save()
            await user.send_message('/gpt /2 Who won the world series in 2020?')

            expected_call_args = [{'role': 'system', 'content': 'this is a test quick system message'},
                                  {'role': 'user', 'content': 'Who won the world series in 2020?'}]
            assert_gpt_api_called_with(mock_method, model='gpt-4-1106-preview', messages=expected_call_args)

    async def test_empty_prompt_after_quick_system_prompt(self):
        chat, user = init_chat_user()
        expected_reply = generate_no_parameters_given_notification_msg()
        await user.send_message('/gpt /1')
        self.assertEqual(expected_reply, chat.last_bot_txt())

    async def test_set_new_quick_system_prompt(self):
        chat, user = init_chat_user()
        await user.send_message('/gpt /1 = new prompt')
        self.assertEqual('Uusi pikaohjausviesti 1 asetettu.', chat.last_bot_txt())
        expected_quick_system_prompts = {'1': 'new prompt'}
        quick_system_prompts = database.get_quick_system_prompts(chat.id)
        self.assertEqual(expected_quick_system_prompts, quick_system_prompts)

    async def test_set_new_quick_system_prompt_can_have_any_amount_of_whitespace_around_equal_sign(self):
        chat, user = init_chat_user()
        await user.send_message('/gpt /1= new prompt two')
        self.assertEqual('Uusi pikaohjausviesti 1 asetettu.', chat.last_bot_txt())
        await user.send_message('/gpt /1 =new prompt two')
        self.assertEqual('Uusi pikaohjausviesti 1 asetettu.', chat.last_bot_txt())
        await user.send_message('/gpt /1=new prompt two')
        self.assertEqual('Uusi pikaohjausviesti 1 asetettu.', chat.last_bot_txt())
        await user.send_message('/gpt /1 = new prompt two')
        self.assertEqual('Uusi pikaohjausviesti 1 asetettu.', chat.last_bot_txt())

    async def test_empty_set_quick_system_message_should_trigger_help_message_if_no_quick_system_message(self):
        openai_api_utils.state.reset_cost_so_far()
        chat, user = init_chat_user()
        await user.send_message('/gpt /1 =')
        expected_reply = 'Nykyinen pikaohjausviesti 1 on nyt tyhjä. ' \
                         'Voit asettaa pikaohjausviestin sisällön komennolla \'/gpt 1 = (uusi viesti)\'.'
        self.assertEqual(expected_reply, chat.last_bot_txt())

    async def test_empty_set_quick_system_message_should_show_existing_quick_system_message(self):
        openai_api_utils.state.reset_cost_so_far()
        chat, user = init_chat_user()
        await user.send_message('/gpt /1 = already saved prompt')
        await user.send_message('/gpt /1 =')
        expected_reply = 'Nykyinen pikaohjausviesti 1 on nyt:' \
                         '\n\nalready saved prompt'
        self.assertEqual(expected_reply, chat.last_bot_txt())

    def test_remove_cost_so_far_notification(self):
        """ Tests, that bot's additional cost information is removed from given string """
        # Singular context
        original_message = ('Abc defg.\n\nKonteksti: 1 viesti. Rahaa paloi: $0.001260, '
                            'rahaa palanut rebootin jälkeen: $0.001260')
        self.assertEqual('Abc defg.', remove_cost_so_far_notification_and_context_info(original_message))

        # Plural context
        original_message = ('Abc defg.\n\nKonteksti: 5 viestiä. Rahaa paloi: $0.001260, '
                            'rahaa palanut rebootin jälkeen: $0.001260')
        self.assertEqual('Abc defg.', remove_cost_so_far_notification_and_context_info(original_message))

    def test_remove_gpt_command_related_text(self):
        """ Tests, that users gpt-command and possible system message parameter is removed """
        self.assertEqual('what?', remove_gpt_command_related_text('/gpt what?'))
        self.assertEqual('what?', remove_gpt_command_related_text('.gpt .1 what?'))
        # Test for cases that are not even supported yet just to make sure the function works as intended
        self.assertEqual('what?', remove_gpt_command_related_text('!gpt !123 what?'))
        self.assertEqual('what?', remove_gpt_command_related_text('!gpt /help /1 /set-value=0 what?'))

    def test_determine_used_model_based_on_command_and_context(self):
        determine = determine_used_model

        self.assertEqual('gpt-3.5-turbo-1106', determine('/gpt3 test', []).name)
        self.assertEqual('gpt-3.5-turbo-1106', determine('/gpt3.5 test', []).name)

        self.assertEqual('gpt-4-1106-preview', determine('/gpt test', []).name)
        # Would not trigger the command, but just to showcase, that default is used for every other case
        self.assertEqual('gpt-4-1106-preview', determine('/gpt3. test', []).name)
        self.assertEqual('gpt-4-1106-preview', determine('/gpt4 test', []).name)

    async def test_correct_model_is_given_in_openai_api_call(self):
        openai_api_utils.state.reset_cost_so_far()
        chat, user = init_chat_user()

        mock_method = AsyncMock()
        mock_method.return_value = get_json(MockOpenAIObject())

        with mock.patch('bobweb.bob.async_http.post_expect_json', mock_method):
            expected_messages = [{'role': 'user', 'content': 'test'}]

            await user.send_message('/gpt test')
            assert_gpt_api_called_with(mock_method, model='gpt-4-1106-preview', messages=expected_messages)
            await user.send_message('/gpt4 test')
            assert_gpt_api_called_with(mock_method, model='gpt-4-1106-preview', messages=expected_messages)

            await user.send_message('/gpt3 test')
            assert_gpt_api_called_with(mock_method, model='gpt-3.5-turbo-1106', messages=expected_messages)
            await user.send_message('/gpt3.5 test')
            assert_gpt_api_called_with(mock_method, model='gpt-3.5-turbo-1106', messages=expected_messages)

    async def test_message_with_image(self):
        """
        Case where user sends a gpt command message with an image and then replies to it with another message.
        Bot should contain same base64 string for the image in both of the requests
        """
        chat, user = init_chat_user()

        mock_method = AsyncMock()
        mock_method.return_value = get_json(MockOpenAIObject())

        with (mock.patch('bobweb.bob.async_http.post_expect_json', mock_method),
              mock.patch('bobweb.bob.telethon_service.client', MockTelethonClientWrapper(chat.bot))):
            photo = (PhotoSize('1', '1', 1, 1, 1),)  # Tuple of PhotoSize objects
            initial_message = await user.send_message('/gpt foo', photo=photo)

            # Now message history list should have the image url in it
            expected_initial_message = {'role': 'user',
                                        'content': [
                                            {'type': 'text', 'text': 'foo'},
                                            {'type': 'image_url',
                                             'image_url': {'url': MockTelethonClientWrapper.mock_image_url}}
                                        ]}
            assert_gpt_api_called_with(mock_method, model='gpt-4-vision-preview', messages=[expected_initial_message])

            # Bots response is now ignored and the user replies to their previous message.
            # Should have same content as previously with the image in the message.
            # Users new message has been added to the history

            await user.send_message('/gpt bar', reply_to_message=initial_message)
            expected_messages = [
                expected_initial_message,  # Same message as previously
                {'role': 'user',
                 'content': [
                     {'type': 'text', 'text': 'bar'}]}
            ]
            assert_gpt_api_called_with(mock_method, model='gpt-4-vision-preview', messages=expected_messages)


    async def test_client_response_gene_error(self):
        chat, user = init_chat_user()
        with mock.patch('bobweb.bob.command_gpt.generate_and_format_result_text', raises_response_generation_exception):
            await user.send_message('/gpt test')

        self.assertIn('response generation raised an exception', chat.last_bot_txt())

    async def test_service_unavailable_error(self):
        chat, user = init_chat_user()
        with mock.patch('bobweb.bob.async_http.post_expect_json', speech_api_mock_response_service_unavailable_error):
            await user.send_message('/gpt test')

        self.assertIn('OpenAi:n palvelu ei ole käytettävissä tai se on juuri nyt ruuhkautunut.',
                      chat.last_bot_txt())

    async def test_rate_limit_error(self):
        chat, user = init_chat_user()
        with mock.patch('bobweb.bob.async_http.post_expect_json', speech_api_mock_response_rate_limit_error_error):
            await user.send_message('/gpt test')

        self.assertIn('OpenAi:n palvelu ei ole käytettävissä tai se on juuri nyt ruuhkautunut.',
                      chat.last_bot_txt())


def get_cost_str(prompt_count: int) -> str:
    return format_money(prompt_count * 0.000940)


def format_money(money: float) -> str:
    return '{:f}'.format(money)
