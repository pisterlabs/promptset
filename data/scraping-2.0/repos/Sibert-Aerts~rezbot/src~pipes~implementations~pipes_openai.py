from configparser import ConfigParser
import discord

from .pipes import pipe_from_class, one_to_many, set_category
from pipes.core.signature import Par, with_signature
from utils.util import parse_bool
import permissions


#####################################################
#                   Pipes : OPENAI                  #
#####################################################
set_category('OPENAI')

_is_openai_usable = False

def openai_setup():
    global openai, _is_openai_usable

    # Attempt import
    try: import openai
    except: return print('Could not import Python module `openai`, OpenAI-related features will not be available.')

    # Attempt to read API key from config
    config = ConfigParser()
    config.read('config.ini')
    openai_api_key = config['OPENAI']['api_key']
    if not openai_api_key or openai_api_key == 'PutYourKeyHere':
        return print('OpenAI API key not set in config.ini, OpenAI-related features will not be available.')
    openai.api_key = openai_api_key

    # Reduce timeout to something slightly more sensible, down from 10 minutes
    import openai.api_requestor
    openai.api_requestor.TIMEOUT_SECS = 60

    _is_openai_usable = True

openai_setup()

@pipe_from_class
class PipeGPTComplete:
    name = 'gpt_complete'
    aliases = ['gpt_extend']
    command = True

    def may_use(user: discord.User):
        return permissions.has(user.id, permissions.trusted)

    @with_signature(
        model             = Par(str, 'ada', 'The GPT model to use, generally: ada/babbage/curie/davinci.'),
        n                 = Par(int, 1, 'The number of completions to generate.', check=lambda n: n <= 10),
        max_tokens        = Par(int, 50, 'The limit of tokens to generate per completion, does not include prompt.'),
        temperature       = Par(float, .7, 'Value between 0 and 2 determining how creative/unhinged the generation is.'),
        presence_penalty  = Par(float, 0, 'Value between -2 and 2, positive values discourage reusing already present words.'),
        frequency_penalty = Par(float, 0, 'Value between -2 and 2, positive values discourage reusing frequently used words.'),
        stop              = Par(str, None, 'String that, if generated, marks the immediate end of the completion.', required=False),
        prepend_prompt    = Par(parse_bool, True, 'Whether to automatically prepend the input prompt to each completion.'),
    )
    @one_to_many
    @staticmethod
    def pipe_function(text, prepend_prompt, **kwargs):
        '''
        Generate a completion to the individual given inputs.
        Uses OpenAI GPT models.
        '''
        if not _is_openai_usable: return [text]

        response = openai.Completion.create(prompt=text, **kwargs)
        completions = [choice.text for choice in response.choices]
        if prepend_prompt:
            completions = [text + completion for completion in completions]
        return completions


@pipe_from_class
class PipeGPTChat:
    name = 'gpt_chat'
    command = True

    def may_use(user: discord.User):
        return permissions.has(user.id, permissions.trusted)

    @with_signature(
        system      = Par(str, None, 'The "system" chat prompt.', required=False),
        user        = Par(str, None, 'An example "user" chat message.', required=False),
        assistant   = Par(str, None, 'An example "assistant" chat response.', required=False),
        # Kwargs
        model             = Par(str, 'gpt-3.5-turbo', 'The GPT model to use, generally: gpt-3.5-turbo/gpt-4.'),
        n                 = Par(int, 1, 'The number of completions to generate.', check=lambda n: n <= 10),
        max_tokens        = Par(int, 50, 'The limit of tokens to generate per completion, does not include prompt.'),
        temperature       = Par(float, .7, 'Value between 0 and 2 determining how creative/unhinged the generation is.'),
        presence_penalty  = Par(float, 0, 'Value between -2 and 2, positive values discourage reusing already present words.'),
        frequency_penalty = Par(float, 0, 'Value between -2 and 2, positive values discourage reusing frequently used words.'),
        stop              = Par(str, None, 'String that, if generated, marks the immediate end of the completion.', required=False),
    )
    @staticmethod
    def pipe_function(items, *, model, system, user, assistant, **kwargs):
        '''
        Generate a chat completion for the given input.
        The given input is treated as a "user" message, and GPT generates an "assistant" response.
        
        You can fill out existing message history by using the system/user/assistant arguments.
        The "system" message will authoritatively declare what the situation is and how the assistant should behave.
        The "user" and "assistant" messages can be used to provide an example of the assistant's response to a user message.
        '''
        if not _is_openai_usable: return items

        base_messages = []
        if system:
            base_messages.append({'role': 'system', 'content': system})
        if user:
            base_messages.append({'role': 'user', 'content': user})
        if assistant:
            base_messages.append({'role': 'assistant', 'content': assistant})

        result = []
        if items:
            for item in items:
                message = {'role': 'user', 'content': item}
                response = openai.ChatCompletion.create(model=model, messages=(base_messages + [message]), **kwargs)
                result.extend(choice.message.content for choice in response.choices)
        else:
            response = openai.ChatCompletion.create(model=model, messages=base_messages, **kwargs)
            result.extend(choice.message.content for choice in response.choices)

        return result
