from chatbot.exceptions.InternalServiceException import InternalServiceException
from chatbot.accessor.DatabaseAccessor import DatabaseAccessor
from chatbot.accessor.OpenAIAccessor import OpenAIAccessor


class BotService:

    def __init__(self):
        self.openAI_accessor = OpenAIAccessor()
        self.database_accessor = DatabaseAccessor()

    def prompt_and_respond(self, question):
        """
        This method will take in a question, extract the context from the database, and then send the context and
        question to the OpenAI API to get a response.

        :param question: The question asked by the user
        :return: The response from the bot answering the question

        """

        try:
            context = self.database_accessor.get_context_from_db(question)

        except InternalServiceException:
            context = "Error: Unable to reach database"
        return self.openAI_accessor.get_response(context, question)








