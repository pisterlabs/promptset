from django.test import TestCase

from openai_integration.openai_chat import Chat


class ChatTestCase(TestCase):
    def setUp(self):
        self.thread_id = None

    def __test_openai_chat(self):
        chat = Chat()
        self.thread_id = chat.thread.id
        chat.create_user_message('Hi')
        run = chat.run_assistant()
        chat.wait_until_run_is_completed(run)

        chat.create_user_message('How are you?')
        run = chat.run_assistant()
        chat.wait_until_run_is_completed(run)
        chat.check_run_status(run)

        chat.create_user_message('I want to subscribe to your service')
        run = chat.run_assistant()
        chat.wait_until_run_is_completed(run)
        status = chat.check_run_status(run)
        print(chat.thread.id)

        self.assertEquals(status.status, 'completed')

    def test_continuing_conversation(self):
        self.__test_openai_chat()
        chat = Chat(self.thread_id)
        chat.create_user_message("I'm fine, thanks")
        run = chat.run_assistant()
        chat.wait_until_run_is_completed(run)
        status = chat.check_run_status(run)
        chat.log_message_content()

        self.assertEquals(status.status, 'completed')
