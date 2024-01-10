import pydantic
import logging
import json
import pprint
import openai
import tempfile
import time
from typing import Optional
from strenum import StrEnum
from asgiref.sync import  sync_to_async
import cloudlanguagetools.chatapi
import cloudlanguagetools.options
import cloudlanguagetools.encryption
from cloudlanguagetools_chatbot import prompts

logger = logging.getLogger(__name__)

class InputType(StrEnum):
    new_sentence = 'NEW_SENTENCE',
    question_or_command = 'QUESTION_OR_COMMAND'
    instructions = 'INSTRUCTIONS'

class CategorizeInputQuery(pydantic.BaseModel):
    input_type: InputType = pydantic.Field(description=prompts.DESCRIPTION_FLD_IS_NEW_QUESTION)
    instructions: Optional[str] = pydantic.Field(description=prompts.DESCRIPTION_FLD_INSTRUCTIONS)


REQUEST_TIMEOUT=15

"""
holds an instance of a conversation
"""
class ChatModel():
    FUNCTION_NAME_TRANSLATE_OR_DICT = 'translate_or_lookup'
    FUNCTION_NAME_TRANSLITERATE = 'transliterate'
    FUNCTION_NAME_BREAKDOWN = 'breakdown'
    FUNCTION_NAME_PRONOUNCE = 'pronounce'

    def __init__(self, manager, audio_format=cloudlanguagetools.options.AudioFormat.mp3):
        self.manager = manager
        self.chatapi = cloudlanguagetools.chatapi.ChatAPI(self.manager)
        self.instruction = prompts.DEFAULT_INSTRUCTIONS
        self.message_history = []
        self.last_call_messages = None
        self.last_input_sentence = None
        self.audio_format = audio_format

        # to use the Azure OpenAI API
        use_azure_openai = True
        if use_azure_openai:
            # configure openai
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling
            # https://github.com/openai/openai-python/issues/517#issuecomment-1645092367
            azure_openai_config = cloudlanguagetools.encryption.decrypt()['OpenAI']
            openai.api_type = "azure"
            openai.api_base = azure_openai_config['azure_endpoint']
            openai.api_version = "2023-07-01-preview"
            openai.api_key = azure_openai_config['azure_api_key']
            self.azure_openai_deployment_name = azure_openai_config['azure_deployment_name']
    
    def set_instruction(self, instruction):
        self.instruction = instruction

    def get_instruction(self):
        return self.instruction

    def get_last_call_messages(self):
        return self.last_call_messages

    def set_send_message_callback(self, send_message_fn, send_audio_fn, send_status_fn):
        self.send_message = send_message_fn
        self.send_audio = send_audio_fn
        self.send_status = send_status_fn

    def get_system_messages(self):
        # do we have any instructions ?
        instruction_message_list = []
        if self.instruction != None:
            instruction_message_list = [{"role": "system", "content": self.instruction}]

        messages = [
            {"role": "system", "content": prompts.SYSTEM_MSG_ASSISTANT},
        ] + instruction_message_list

        return messages

    async def call_openai(self):

        messages = self.get_system_messages()
        messages.extend(self.message_history)

        self.last_call_messages = messages

        logger.debug(f"sending messages to openai: {pprint.pformat(messages)}")

        response = await openai.ChatCompletion.acreate(
            # for OpenAI:
            # model="gpt-3.5-turbo-0613"
            # for Azure:
            engine=self.azure_openai_deployment_name,
            messages=messages,
            functions=self.get_openai_functions(),
            function_call= "auto",
            temperature=0.0,
            request_timeout=REQUEST_TIMEOUT
        )

        return response

    async def categorize_input_type(self, last_input_sentence, input_sentence) -> InputType:
        """return true if input is a new sentence. we'll use this to clear history"""

        if last_input_sentence != None:
            last_input_sentence_entry = [{"role": "user", "content": last_input_sentence}]
        else:
            last_input_sentence_entry = [{"role": "user", "content": "There is no previous sentence"}]

        messages = [
            {"role": "system", "content": prompts.SYSTEM_MSG_ASSISTANT}
        ] + last_input_sentence_entry + [
            {"role": "user", "content": input_sentence}
        ]

        categorize_input_type_name = 'category_input_type'

        response = await openai.ChatCompletion.acreate(
            # for OpenAI:
            # model="gpt-3.5-turbo-0613"
            # for Azure:            
            engine=self.azure_openai_deployment_name,
            messages=messages,
            functions=[{
                'name': categorize_input_type_name,
                'description': prompts.DESCRIPTION_FN_IS_NEW_QUESTION,
                'parameters': CategorizeInputQuery.model_json_schema(),
            }],
            function_call={'name': categorize_input_type_name},
            temperature=0.0,
            request_timeout=REQUEST_TIMEOUT
        )

        message = response['choices'][0]['message']
        function_name = message['function_call']['name']
        assert function_name == categorize_input_type_name
        logger.debug(f'categorize_input_type response: {pprint.pformat(message)}')
        arguments = json.loads(message["function_call"]["arguments"])
        input_type_result = CategorizeInputQuery(**arguments)
        
        logger.info(f'input sentence: [{input_sentence}] input type: {input_type_result}')
        return input_type_result

    async def process_audio(self, audio_tempfile: tempfile.NamedTemporaryFile):
        async_recognize_audio = sync_to_async(self.chatapi.recognize_audio)
        text = await async_recognize_audio(audio_tempfile, self.audio_format)
        await self.send_status(f'recognized text: {text}')
        await self.process_message(text)

    async def process_instructions(self, instructions):
        self.set_instruction(instructions)
        await self.send_status(f'My instructions are now: {self.get_instruction()}')

    async def process_message(self, input_message):
    
        input_type_result = await self.categorize_input_type(self.last_input_sentence, input_message)
        if input_type_result.input_type == InputType.new_sentence:
            # user is moving on to a new sentence, clear history
            self.message_history = []
            self.last_input_sentence = input_message
        elif input_type_result.input_type == InputType.question_or_command:
            # user has a question about previous sentence, don't clear history
            pass
        elif input_type_result.input_type == InputType.instructions:
            return await self.process_instructions(input_type_result.instructions)

        max_calls = 10
        continue_processing = True

        # message_history contains the most recent request
        self.message_history.append({"role": "user", "content": input_message})


        function_call_cache = {}
        at_least_one_message_to_user = False

        try:
            while continue_processing and max_calls > 0:
                max_calls -= 1
                response = await self.call_openai()
                logger.debug(pprint.pformat(response))
                message = response['choices'][0]['message']
                message_content = message.get('content', None)
                if 'function_call' in message:
                    function_name = message['function_call']['name']
                    logger.info(f'function_call: function_name: {function_name}')
                    try:
                        arguments = json.loads(message["function_call"]["arguments"])
                    except json.decoder.JSONDecodeError as e:
                        logger.exception(f'error decoding json: {message}')
                    arguments_str = json.dumps(arguments, indent=4)
                    # check whether we've called that function with exact same arguments before
                    if arguments_str not in function_call_cache.get(function_name, {}):
                        # haven't called it with these arguments before
                        function_call_result, sent_message_to_user = await self.process_function_call(function_name, arguments)
                        at_least_one_message_to_user = at_least_one_message_to_user or sent_message_to_user
                        self.message_history.append({"role": "function", "name": function_name, "content": function_call_result})
                        # cache function call results
                        if function_name not in function_call_cache:
                            function_call_cache[function_name] = {}
                        function_call_cache[function_name][arguments_str] = function_call_result
                    else:
                        # we've called that function already with same arguments, we won't call again, but
                        # add to history again, so that chatgpt doesn't call the function again
                        self.message_history.append({"role": "function", "name": function_name, "content": function_call_result})
                else:
                    continue_processing = False
                    if at_least_one_message_to_user == False:
                        # or nothing has been shown to the user yet, so we should show the final message. maybe chatgpt is trying to explain something
                        await self.send_message(message['content'])
                
                # if there was a message, append it to the history
                if message_content != None:
                    self.message_history.append({"role": "assistant", "content": message_content})
        except Exception as e:
            logger.exception(f'error processing function call')
            await self.send_status(f'error: {str(e)}')


    async def process_function_call(self, function_name, arguments):
        # by default, don't send output to user
        send_message_to_user = False
        if function_name == self.FUNCTION_NAME_PRONOUNCE:
            query = cloudlanguagetools.chatapi.AudioQuery(**arguments)
            async_audio = sync_to_async(self.chatapi.audio)
            audio_tempfile = await async_audio(query, self.audio_format)
            result = query.input_text
            await self.send_audio(audio_tempfile)
            send_message_to_user = True
        else:
            # text-based functions
            try:
                if function_name == self.FUNCTION_NAME_TRANSLATE_OR_DICT:
                    translate_query = cloudlanguagetools.chatapi.TranslateLookupQuery(**arguments)
                    async_translate = sync_to_async(self.chatapi.translate_or_lookup)
                    result = await async_translate(translate_query)
                    send_message_to_user = True
                elif function_name == self.FUNCTION_NAME_TRANSLITERATE:
                    query = cloudlanguagetools.chatapi.TransliterateQuery(**arguments)
                    async_transliterate = sync_to_async(self.chatapi.transliterate)
                    result = await async_transliterate(query)
                    send_message_to_user = True
                elif function_name == self.FUNCTION_NAME_BREAKDOWN:
                    query = cloudlanguagetools.chatapi.BreakdownQuery(**arguments)
                    async_breakdown = sync_to_async(self.chatapi.breakdown)
                    result = await async_breakdown(query)
                    send_message_to_user = True
                else:
                    # report unknown function
                    result = f'unknown function: {function_name}'
            except cloudlanguagetools.chatapi.NoDataFoundException as e:
                result = str(e)
            logger.info(f'function: {function_name} result: {result}')
            if send_message_to_user:
                await self.send_message(result)
        # need to echo the result back to chatgpt
        return result, send_message_to_user    

    def get_openai_functions(self):
        return [
            {
                'name': self.FUNCTION_NAME_TRANSLATE_OR_DICT,
                'description': "Translate or do a dictionary lookup for input text from source language to target language",
                'parameters': cloudlanguagetools.chatapi.TranslateLookupQuery.model_json_schema(),
            },
            {
                'name': self.FUNCTION_NAME_TRANSLITERATE,
                'description': "Transliterate the input text in the given language. This can be used for Pinyin or Jyutping for Chinese, or Romaji for Japanese",
                'parameters': cloudlanguagetools.chatapi.TransliterateQuery.model_json_schema(),
            },
            {
                'name': self.FUNCTION_NAME_BREAKDOWN,
                'description': "Breakdown the given sentence into words",
                'parameters': cloudlanguagetools.chatapi.BreakdownQuery.model_json_schema(),
            },            
            {
                'name': self.FUNCTION_NAME_PRONOUNCE,
                'description': "Pronounce input text in the given language (generate text to speech audio)",
                'parameters': cloudlanguagetools.chatapi.AudioQuery.model_json_schema(),
            },
        ]
