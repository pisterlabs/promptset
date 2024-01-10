import datetime
import io

from unittest import mock

import django
import pytest
from django.core import management
from django.test import TestCase
from unittest.mock import patch

from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from openai import InvalidRequestError
from openai.openai_response import OpenAIResponse

import bobweb.bob.config
from bobweb.bob import main, image_generating_service, async_http
from bobweb.bob.image_generating_service import convert_base64_strings_to_images, get_3x3_image_compilation, \
    ImageGeneratingModel
from bobweb.bob.resources.test.openai_api_dalle_images_response_dummy import openai_dalle_create_request_response_mock
from bobweb.bob.test_command_gpt import mock_response_from_openai
from bobweb.bob.tests_mocks_v2 import init_chat_user, MockUpdate, MockMessage
from bobweb.bob.tests_utils import assert_reply_to_contain, \
    assert_reply_equal, assert_get_parameters_returns_expected_value, \
    assert_command_triggers, mock_request_raises_client_response_error

from bobweb.bob.command_image_generation import send_images_response, get_image_file_name, DalleMiniCommand, \
    DalleCommand, get_text_in_html_str_italics_between_quotes, remove_all_dalle_and_dallemini_commands_related_text
from bobweb.bob.resources.test.dallemini_images_base64_dummy import base64_mock_images


# Simple test that images are similar. Reduces images to be 100 x 100 and then compares contents
# Reason for similarity check is that image saved to disk is not identical to a new image generated. That's why
# we need to conclude that the images are just close enough similar.
# based on: https://rosettacode.org/wiki/Percentage_difference_between_images#Python
def assert_images_are_similar_enough(test_case, img1, img2):
    img1 = img1.resize((256, 256))
    img2 = img2.resize((256, 256))
    assert img1.mode == img2.mode, "Different kinds of images."

    pairs = zip(img1.getdata(), img2.getdata())
    if len(img1.getbands()) == 1:
        # for gray-scale jpegs
        dif = sum(abs(p1 - p2) for p1, p2 in pairs)
    else:
        dif = sum(abs(c1 - c2) for p1, p2 in pairs for c1, c2 in zip(p1, p2))

    ncomponents = img1.size[0] * img1.size[1] * 3
    actual_percentage_difference = (dif / 255.0 * 100) / ncomponents
    tolerance_percentage = 1
    test_case.assertLess(actual_percentage_difference, tolerance_percentage)


# # Test that images are similar enough using imagehash package
# import imagehash
# def assert_images_are_similar_enough(self, image1, image2):
#     hash1 = imagehash.average_hash(image1)
#     hash2 = imagehash.average_hash(image2)
#     hash_bit_difference = hash1 - hash2
#     tolerance = 5  # maximum bits that could be different between the hashes.
#     self.assertLess(hash_bit_difference, tolerance)


async def assert_send_image_response(test_case):
    chat, user = init_chat_user()
    message = MockMessage(chat, user)
    update = MockUpdate(message=message)
    caption = get_text_in_html_str_italics_between_quotes('test')
    await send_images_response(update, caption, [test_case.expected_image_result])

    # Message text should be in quotes and in italics
    test_case.assertEqual('"<i>test</i>"', chat.last_bot_txt())

    actual_image_bytes = chat.media_and_documents[-1]
    actual_image_stream = io.BytesIO(actual_image_bytes)
    actual_image = Image.open(actual_image_stream)

    # make sure that the image looks like expected
    assert_images_are_similar_enough(test_case, test_case.expected_image_result, actual_image)


async def assert_command_reply_to_message_with_content_results_to_prompt(test_case, expected_prompt, original_message):
    chat, user = init_chat_user()
    message = await user.send_message(original_message)

    with mock.patch('bobweb.bob.image_generating_service.generate_images',
                    wraps=bobweb.bob.image_generating_service.generate_images) as mock_generate_images:
        # Now when user replies to another message with only the command,
        # it should use the other message as the prompt
        await user.send_message('/dalle', reply_to_message=message)
        test_case.assertIn(f'"<i>{expected_prompt}</i>"', chat.last_bot_txt())
        mock_generate_images.assert_called_once_with(expected_prompt, model=ImageGeneratingModel.DALLE2)


async def dallemini_mock_response_200_with_base64_images(*args, **kwargs):
    return str.encode(f'{{"images": {base64_mock_images},"version":"mega-bf16:v0"}}\n')


async def openai_api_mock_response_one_image(*args, **kwargs):
    return OpenAIResponse(openai_dalle_create_request_response_mock['data'], None)


async def raise_safety_system_triggered_error(*args, **kwargs):
    raise InvalidRequestError(message='Your request was rejected as a result of our safety system. Your prompt '
                                      'may contain text that is not allowed by our safety system.', param=None)


@pytest.mark.asyncio
@mock.patch('bobweb.bob.async_http.post_expect_bytes', dallemini_mock_response_200_with_base64_images)
class DalleminiCommandTests(django.test.TransactionTestCase):
    command_class = DalleMiniCommand
    command_str = 'dallemini'
    expected_image_result: Image = Image.open('bobweb/bob/resources/test/test_get_3x3_image_compilation-expected.jpeg')

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        management.call_command('migrate')

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        if cls.expected_image_result:
            cls.expected_image_result.close()
        async_http.client.close()

    async def test_command_triggers(self):
        should_trigger = [f'/{self.command_str}', f'!{self.command_str}', f'.{self.command_str}',
                          f'/{self.command_str.upper()}', f'/{self.command_str} test']
        should_not_trigger = [f'{self.command_str}', f'test /{self.command_str}']
        await assert_command_triggers(self, self.command_class, should_trigger, should_not_trigger)

    async def test_no_prompt_gives_help_reply(self):
        await assert_reply_equal(self, f'/{self.command_str}',
                                 "Anna jokin syöte komennon jälkeen. '[.!/]prompt [syöte]'")

    async def test_reply_contains_given_prompt_in_italics_and_quotes(self):
        await assert_reply_to_contain(self, f'/{self.command_str} 1337', ['"<i>1337</i>"'])

    async def test_get_given_parameter(self):
        assert_get_parameters_returns_expected_value(self, f'!{self.command_str}', self.command_class())

    async def test_send_image_response(self):
        await assert_send_image_response(self)

    async def test_convert_base64_strings_to_images(self):
        images = convert_base64_strings_to_images(base64_mock_images)
        self.assertEqual(len(images), 9)
        self.assertEqual(type(images[0]), JpegImageFile)

    async def test_get_image_compilation_file_name(self):
        with patch('bobweb.bob.command_image_generation.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(1970, 1, 1, 1, 1)

            non_valid_name = '!"#¤%&/()=?``^*@£$€{[]}`\\~`` test \t \n foo-_b.a.r.jpeg'
            expected = '1970-01-01_0101_dalle_mini_with_prompt_test-foo-_barjpeg.jpeg'
            self.assertEqual(expected, get_image_file_name(non_valid_name))

    async def test_converted_3x3_image_compilation_is_similar_to_expected(self):
        images = convert_base64_strings_to_images(base64_mock_images)
        actual_image_obj = get_3x3_image_compilation(images)

        # Test dimensions to match
        expected_width = images[0].width * 3
        expected_height = images[0].height * 3
        self.assertEqual(expected_width, actual_image_obj.width, '3x3 image compilation width does not match')
        self.assertEqual(expected_height, actual_image_obj.height, '3x3 image compilation height does not match')

        # make sure that the image looks like expected
        assert_images_are_similar_enough(self, self.expected_image_result, actual_image_obj)

    async def test_response_status_not_200_gives_error_msg(self):
        with mock.patch('bobweb.bob.async_http.post_expect_bytes', mock_request_raises_client_response_error()):
            await assert_reply_to_contain(self,
                                          f'/{self.command_str} 1337',
                                          ['Kuvan luominen epäonnistui. Lisätietoa Bobin lokeissa.'])


@pytest.mark.asyncio
@mock.patch('openai.Image.acreate', openai_api_mock_response_one_image)
@mock.patch('bobweb.bob.openai_api_utils.user_has_permission_to_use_openai_api', lambda *args: True)
class DalleCommandTests(django.test.TransactionTestCase):
    bobweb.bob.config.openai_api_key = 'DUMMY_VALUE_FOR_ENVIRONMENT_VARIABLE'
    command_class = DalleCommand
    command_str = 'dalle'
    expected_image_result: Image = Image.open(
        'bobweb/bob/resources/test/openai_api_dalle_images_response_processed_image.jpg')

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        management.call_command('migrate')

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        if cls.expected_image_result:
            cls.expected_image_result.close()
        async_http.client.close()

    async def test_command_triggers(self):
        should_trigger = [f'/{self.command_str}', f'!{self.command_str}', f'.{self.command_str}',
                          f'/{self.command_str.upper()}', f'/{self.command_str} test']
        should_not_trigger = [f'{self.command_str}', f'test /{self.command_str}']
        await assert_command_triggers(self, self.command_class, should_trigger, should_not_trigger)

    async def test_no_prompt_gives_help_reply(self):
        await assert_reply_equal(self, f'/{self.command_str}',
                                 "Anna jokin syöte komennon jälkeen. '[.!/]prompt [syöte]'")

    @mock.patch('openai.ChatCompletion.acreate', mock_response_from_openai)
    async def test_known_openai_api_commands_and_price_info_is_removed_from_replied_messages_when_reply(self):
        expected_cases = [
            ('something', 'something'),
            ('something', '"<i>something</i>"'),
            ('Abc', 'Abc\n\nKonteksti: 1 viesti. Rahaa paloi: $0.001260, rahaa palanut rebootin jälkeen: $0.001260'),
            ('Abc', 'Abc\n\nRahaa paloi: $0.001260, rahaa palanut rebootin jälkeen: $0.001260'),
            ('Abc', '/gpt /1 Abc')
        ]
        for case in expected_cases:
            await assert_command_reply_to_message_with_content_results_to_prompt(
                self, case[0], case[1])

    def test_all_dalle_related_text_is_removed(self):
        self.assertEqual('test', remove_all_dalle_and_dallemini_commands_related_text('/dalle test'))
        self.assertEqual('test', remove_all_dalle_and_dallemini_commands_related_text('/dallemini test'))
        self.assertEqual('test', remove_all_dalle_and_dallemini_commands_related_text('/dalle /dallemini test'))
        self.assertEqual('/abc test', remove_all_dalle_and_dallemini_commands_related_text('/dalle /dallemini /abc test'))
        self.assertEqual('', remove_all_dalle_and_dallemini_commands_related_text('/dallemini'))

    async def test_reply_contains_given_prompt_in_italics_and_quotes(self):
        await assert_reply_to_contain(self, f'/{self.command_str} 1337', ['"<i>1337</i>"'])

    async def test_get_given_parameter(self):
        assert_get_parameters_returns_expected_value(self, f'!{self.command_str}', self.command_class())

    async def test_send_image_response(self):
        await assert_send_image_response(self)

    async def test_convert_base64_strings_to_images(self):
        images = convert_base64_strings_to_images(base64_mock_images)
        self.assertEqual(len(images), 9)
        self.assertEqual(type(images[0]), JpegImageFile)

    async def test_get_image_compilation_file_name(self):
        with patch('bobweb.bob.command_image_generation.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(1970, 1, 1, 1, 1)

            non_valid_name = '!"#¤%&/()=?``^*@£$€{[]}`\\~`` test \t \n foo-_b.a.r.jpeg'
            expected = '1970-01-01_0101_dalle_mini_with_prompt_test-foo-_barjpeg.jpeg'
            self.assertEqual(expected, get_image_file_name(non_valid_name))

    async def test_multiple_context_managers_and_asserting_raised_exception(self):
        """ More example than tests. Demonstrates how context manager can contain multiple definitions and confirms
            that actual api is not called """
        with (
            mock.patch('openai.Image.acreate', raise_safety_system_triggered_error),
            self.assertRaises(InvalidRequestError) as e,
        ):
            await image_generating_service.generate_images('test prompt', ImageGeneratingModel.DALLE2)

    async def test_bot_gives_notification_if_safety_system_error_is_triggered(self):
        with mock.patch('openai.Image.acreate', raise_safety_system_triggered_error):
            chat, user = init_chat_user()
            await user.send_message('/dalle inappropriate prompt that should raise error')
            self.assertEqual(DalleCommand.safety_system_error_msg, chat.last_bot_txt())

    async def test_image_sent_by_bot_is_similar_to_expected(self):
        chat, user = init_chat_user()
        await user.send_message('/dalle some prompt')

        image_bytes_sent_by_bot = chat.media_and_documents[-1]
        actual_image: Image = Image.open(io.BytesIO(image_bytes_sent_by_bot))

        # make sure that the image looks like expected
        assert_images_are_similar_enough(self, self.expected_image_result, actual_image)

    async def test_user_has_no_permission_to_use_api_gives_notification(self):
        with mock.patch('bobweb.bob.openai_api_utils.user_has_permission_to_use_openai_api', lambda *args: False):
            chat, user = init_chat_user()
            await user.send_message('/dalle whatever')
            self.assertEqual('Komennon käyttö on rajattu pienelle testiryhmälle käyttäjiä', chat.last_bot_txt())
