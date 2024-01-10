import openai
import logging
import boto3
import os
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.utils import is_intent_name, get_slot_value
from ask_sdk_model import Response
from ask_sdk_model.ui import SimpleCard
from ask_sdk_model.ui import Image
from ask_sdk_model.ui import StandardCard

# Configure the OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Initialize the AWS SDK
s3 = boto3.client('s3')
bucket_name = "YOUR_S3_BUCKET_NAME"

# Define the logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define the request handlers
class GetGreetingHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("GetGreetingIntent")(handler_input)

    def handle(self, handler_input):
        speech_text = "Olá, seja bem vindo senhor"

        handler_input.response_builder.speak(speech_text).set_card(
            SimpleCard("GPT-3 Skill", speech_text)).set_should_end_session(
            False)
        return handler_input.response_builder.response

class AskQuestionHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AskQuestionIntent")(handler_input)

    def handle(self, handler_input):
        # Get the user's question from the intent
        question = get_slot_value(handler_input=handler_input, slot_name="question")
        
        # Send the question to GPT-3 for processing
        response = openai.Completion.create(engine="davinci", prompt=question, max_tokens=200)
        
        # Extract the answer from GPT-3's response
        answer = response.choices[0].text.strip()
        
        # Save the question and answer to S3
        object_key = "questions/" + question.replace(" ", "_") + ".txt"
        s3.put_object(Body=answer.encode('utf-8'), Bucket=bucket_name, Key=object_key)
        
        # Respond to the user with the answer
        speech_text = "Here is what I found: " + answer
        handler_input.response_builder.speak(speech_text).set_card(
            StandardCard("GPT-3 Skill", question, Image(
                small_image_url="https://s3.amazonaws.com/YOUR_S3_BUCKET_NAME/gpt3.png",
                large_image_url="https://s3.amazonaws.com/YOUR_S3_BUCKET_NAME/gpt3.png"
            ), answer)).set_should_end_session(
            False)
        return handler_input.response_builder.response

# Define the exception handlers
class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)

        speech_text = "Desculpe, tive um erro interno. Tente novamente daqui a pouco senhor."
        handler_input.response_builder.speak(speech_text).set_card(
            SimpleCard("GPT-3 Skill", speech_text)).set_should_end_session(
            True)
        return handler_input.response_builder.response

# Build the Skill
sb = SkillBuilder()

sb.add_request_handler(GetGreetingHandler())
sb.add_request_handler(AskQuestionHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

# Lambda function entry point
def lambda_handler(event, context):
    sb_lambda = sb.lambda_handler()
    return sb_lambda(event, context)