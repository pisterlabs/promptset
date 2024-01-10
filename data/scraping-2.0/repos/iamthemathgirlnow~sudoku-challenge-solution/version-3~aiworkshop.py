from openai import OpenAI
import time
import tiktoken

client = OpenAI()
gpt3_5 = 'gpt-3.5-turbo'
gpt3 = gpt3_5
gpt3_5_16k = 'gpt-3.5-turbo-16k-0613'
gpt4 ='gpt-4-1106-preview'
default_model = gpt3_5

SECONDS_TO_WAIT = 0.5

model_strings = [
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0301',
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-16k-0613',
    'gpt-3.5-turbo-instruct-0914',
    'gpt-4',
    'gpt-4-0314',
    'gpt-4-0613',
    'gpt-4-1106-preview',
    ]

def _get_model_from_string(model: str|None, model_name: str|None) -> tuple[str, str]:
    # Corrects for common ways of writing the model name
    if model == 'gpt3.5' or model == 'gpt-3.5':
        model = gpt3_5
    elif model == 'gpt3.5-16k' or model == 'gpt-3.5-16k':
        model = gpt3_5_16k
    elif model == 'gpt4' or model == 'gpt-4':
        model = gpt4

    # If there is a custom model_name, uses that instead of the model string
    if model is None and model_name is None:
        return default_model, str(default_model)
    elif model is None and model_name is not None:
        if model_name in model_strings:
            return default_model, str(default_model)
        else:
            return default_model, model_name
    elif model is not None and model_name is None:
        return model, str(model)
    elif model is not None and model_name is not None:
        if model_name in model_strings:
            return model, str(model)
        else:
            return model, model_name
    else:
        raise ValueError("Something went wrong in get_model_from_string()")


def get_tokens(string, model = default_model):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_settings(
        model: str|None = None,
        max_tokens: int|None = None,
        temperature: float|None = None,
        seconds_to_wait: float|None = None,
        stream: bool|None = None,
        print_enabled: bool|None = None,
        users_name: str|None = None,
        model_name: str|None = None,
        system_message: str|None = None,
        **kwargs,
        ) -> dict:

    model, model_name = _get_model_from_string(model, model_name)
    max_tokens = max_tokens or None
    temperature = temperature or 0.7
    seconds_to_wait = seconds_to_wait or SECONDS_TO_WAIT
    stream = True if stream is None else stream
    print_enabled = True if print_enabled is None else print_enabled
    users_name = users_name or "User"
    system_message = system_message or None

    default_settings = {
        'model': model,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'seconds_to_wait': seconds_to_wait,
        'stream': stream,
        'print_enabled': print_enabled,
        'users_name': users_name,
        'model_name': model_name,
        'system_message': system_message,
    }
    
    return default_settings


def _setup_settings(**kwargs):
    settings = kwargs.get('settings', None)
    if settings is None:
        settings = get_settings(**kwargs)
    for key, value in kwargs.items():
        settings[key] = value
    return settings


class ChatGPTAPI:
    def __init__(
            self,
            **kwargs):
        
        self.settings = _setup_settings(**kwargs)

    def _fetch_openai_chat_response(
            self,
            messages):
        
        optional_parameters = []
        for key in optional_parameters:
            if key in self.settings and self.settings[key] is None:
                optional_parameters.remove(key)

        api_parameters = {key: self.settings[key] for key in optional_parameters if key in self.settings}

        api_parameters['model'] = self.settings['model']
        api_parameters['messages'] = messages
        api_parameters['temperature'] = self.settings['temperature']
        api_parameters['max_tokens'] = self.settings['max_tokens']
        api_parameters['stream'] = self.settings['stream']

        # Makes the API call with the unpacked parameters
        response = client.chat.completions.create(**api_parameters)

        # # The old version
        # response = openai.ChatCompletion.create(
        #     model=self.settings['model'],
        #     messages=messages,
        #     temperature=self.settings['temperature'],
        #     max_tokens=self.settings['max_tokens'],
        #     stream=self.settings['stream'],
        #     )
        
        time.sleep(self.settings['seconds_to_wait'])
        return response
    

    def _print_users_message(self, messages):
        users_name = self.settings['users_name']
        model_name = self.settings['model_name']
        print_enabled = self.settings['print_enabled']

        if print_enabled:
            print('\n' + users_name + ': \n' + messages[-1]['content'] + '\n')
            print(model_name + ':')


    def _get_stream_from_chat_response(
            self,
            response_stream):
        
        print_enabled = self.settings['print_enabled']
        print_in_real_time = self.settings['print_enabled']

        def get_message_from_response(response):
            if 'message' in response.choices[0]:
                return response.choices[0].message.content
            elif 'delta' in response.choices[0]:
                if 'content' in response.choices[0].delta:
                    return response.choices[0].delta.content
                else:
                    return None
            else: 
                return None

        def return_if_message_exists(data):
            message = get_message_from_response(data)
            if message is not None:
                return message

        entire_message = []

        if self.settings['stream'] == False:
            message = response_stream.choices[0].message.content
            if print_enabled:
                print(message)
            return message, response_stream
        
        last_data = None
        for data in response_stream:
            message = data.choices[0].delta.content or ""
            last_data = data
            if message is None or message == "":
                continue
            entire_message.append(message)
            if print_enabled and print_in_real_time:
                print(message, end="")
        if print_enabled and not print_in_real_time:
            print("".join(entire_message))
        response_message = "".join(entire_message)

        if print_enabled:
            print('\n')
        return response_message, last_data
    

    def format_message(
            self,
            role,
            message):
        
        if isinstance(message, str):
            message = {'role': role, 'content': message}
        return message
        

    def get_response(
            self,
            messages):
        
        if isinstance(self.settings['system_message'], str):
            self.settings['system_message'] = self.format_message('system', self.settings['system_message'])
        if self.settings['system_message'] is not None:
            messages.insert(0, self.settings['system_message'])

        if isinstance(messages, str):
            messages = [self.format_message('user', messages)]
        elif isinstance(messages, dict):
            messages = [messages]
        elif isinstance(messages, list) and all([isinstance(message, dict) for message in messages]):
            pass
        else:
            raise TypeError(f"Tried to pass {type(messages)} to get_response.")
        
        self._print_users_message(messages)
        response_stream = self._fetch_openai_chat_response(messages)
        response_message, response = self._get_stream_from_chat_response(response_stream)
        return response_message, response


class ChatSession:
    def __init__(
            self,
            include_response_object=None,
            **kwargs):

        self.llm = ChatGPTAPI(**kwargs)
        message_history = kwargs.get('message_history', [])
        if not isinstance(message_history, list):
            message_history = [message_history]
        self.message_history = message_history
        self.include_response_object = include_response_object or False


    def _call_api(self, message):
        if isinstance(message, str):
            message = self.llm.format_message('user', message)
        elif isinstance(message, dict):
            pass
        else:
            raise TypeError("ChatSession.send_message() must be passed a str or a dict.")
        
        self.message_history.append(message)
        message_response, response = self.llm.get_response(self.message_history)
        self.message_history.append(self.llm.format_message('assistant', message_response))
        return message_response, response

    def send_message(self, messages: str|dict|list) -> str|list[str]:
        sent_as_list = True
        if isinstance(messages, str) or isinstance(messages, dict) or messages is None:
            sent_as_list = False
            messages = [messages]
        elif isinstance(messages, list):
            pass
        else:
            raise TypeError("ChatSession.send_messages() must be passed a str, a dict, or a list of str or dict.")
        
        message_responses = []
        responses = []
        for message in messages:
            message_response, response = self._call_api(message)
            message_responses.append(message_response)
            responses.append(response)

        if sent_as_list == False and len(message_responses) == 1:
            if not self.include_response_object:
                return message_responses[0]
            return message_responses[0], responses[0]
        
        if not self.include_response_object:
            return message_responses
        return message_responses, responses
    
    def send_messages(self, messages: str|dict|list):
        return self.send_message(messages)


# # Tests that it works. Produces 10 response tokens, enough for the first line of a poem.
# chat = ChatSession(print_enabled=True, model=gpt4, max_tokens=10, temperature=0, stream=True, seed=1, include_response_object=True, system_message="You write brief four line poems. Do not add anything else. Do not add any surrounding commentary, just your poem.")
# message, response = chat.send_message("""Write a four line poem about shoes.""")

