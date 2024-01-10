import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from typing import Dict, List, Callable, Optional, Union
from pydantic import BaseModel
from agentx.schema import Message, Content, ToolCall, FunctionCall, GenerationConfig

OEPNAI_API_KW = [
    'model',
    'frequency_penalty',
    'logit_bias',
    'max_tokens',
    'n',
    'presence_penalty',
    'response_format',
    'seed',
    'stop',
    'temperature',
    'tool_choice',
    'tools',
    'top_p'
]

def add_name(message:Dict, name:Optional[str]=None) -> Dict:
    if name != None:
        message['name'] = name
    return message

def transform_message_openai(message:Message) -> Dict:

    if message.role == 'system':
        return add_name(
            {
                'role': 'system',
                'content': message.content.text,
                },
            message.name
        )

    if message.role == 'user':
        content = message.content
        if content.files == None and content.urls == None:
            return add_name(
                {
                    'role': 'user',
                    'content': content.text,
                },
                message.name
            )
        _content = []
        if content.text != None:
            _content.append({
                'type':'text',
                'text': content.text
            })
        if content.files != None:
            for file in content.files:
                _content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:{file.mime_type};base64,{file.base64Str}'
                    }
                })
        if content.urls != None:
            for url in content.urls:
                _content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': url.unicode_string()
                    }
                })
        return add_name(
            {
                'role':'user',
                'content': _content,
            }, 
            message.name
        )

    if message.role == 'assistant':
        tool_calls = message.content.tool_calls
        if tool_calls == None:
            return add_name(
                {
                    'role': 'assistant',
                    'content': message.content.text,
                }, 
                message.name
            )
        
        _tool_calls = []
        for tool_call in tool_calls:
            _tool_calls.append({
                'id': tool_call.id,
                'type': tool_call.type,
                'function': {
                    'name': tool_call.function_call.name,
                    'arguments': tool_call.function_call.arguments
                }
            })
        return add_name(
            {
                'role': 'assistant',
                'content':'',
                'tool_calls': _tool_calls,
            },
            message.name
        )
    
    if message.role == 'tool':
        return {
            'role': 'tool',
            'content': message.content.tool_response.content,
            'tool_call_id': message.content.tool_response.id,
        }


class OAIClient():
    def __init__(self, generation_config:GenerationConfig):
        api_type = generation_config.api_type
        if api_type == 'openai' or api_type == 'fastchat':
            self.client = openai.OpenAI(
                api_key=generation_config.api_key,
                organization=generation_config.organization,
                base_url=generation_config.base_url.unicode_string(),
                timeout=generation_config.timeout,
                max_retries=generation_config.max_retries,
            )
            self.a_client = openai.AsyncOpenAI(
                api_key=generation_config.api_key,
                organization=generation_config.organization,
                base_url=generation_config.base_url.unicode_string(),
                timeout=generation_config.timeout,
                max_retries=generation_config.max_retries,
            )
        if api_type == 'azure':
            self.client = openai.AzureOpenAI(
                api_key=generation_config.api_key,
                organization=generation_config.organization,
                azure_endpoint=generation_config.base_url.unicode_string(),
                api_version=generation_config.api_version,
                timeout=generation_config.timeout,
                max_retries=generation_config.max_retries,
            )
            self.a_client = openai.AsyncAzureOpenAI(
                api_key=generation_config.api_key,
                organization=generation_config.organization,
                azure_endpoint=generation_config.base_url.unicode_string(),
                api_version=generation_config.api_version,
                timeout=generation_config.timeout,
                max_retries=generation_config.max_retries,
            )
        else:
            raise Exception('Invalid API type.')
    
    def generate(
            self,
            messages: List[Message],
            generation_config: GenerationConfig,
            reduce_function: Optional[Callable[[List[Message]], Message]]=None,
            output_model:BaseModel = None,
    ) -> Union[Message, List[Message]]:
        # call the openai api
        kw_args = {key:value for key, value in generation_config.model_dump().items() if value != None and key in OEPNAI_API_KW}

        if output_model != None:
            messages.append(
                Message(
                    role='user',
                    content=Content(
                        text='You must return a JSON object following this json schema. {schema}'.format(
                            schema = output_model.model_json_schema()
                        )
                    )
                )
            )
            kw_args['response_format'] = {'type':'json_object'}
        
        _messages = []

        num_retry = 0

        if generation_config.tools != None:
            kw_args['tools'] = [
                {
                    'type':'function',
                    'function':tool.model_dump()
                } for tool in generation_config.tools.values()
            ]
        
        if generation_config.api_type == 'azure':
            kw_args['model'] = generation_config.azure_deployment

        while num_retry < generation_config.max_retries:
            response:ChatCompletion = self.client.chat.completions.create(
                messages=[transform_message_openai(message) for message in messages],
                **kw_args
            )

            generated_messages = [choice.message for choice in response.choices]

            for message in generated_messages:
                if message.tool_calls != None:
                    # deterministic ordering of tool calls
                    tool_calls:List[ChatCompletionMessageToolCall] = sorted(message.tool_calls, key=lambda x: x.function.name)
                    
                    content = Content(
                        tool_calls=[
                            ToolCall(
                                id=tool_call.id,
                                type=tool_call.type, 
                                function_call=FunctionCall(
                                    name=tool_call.function.name.lower(), 
                                    arguments=tool_call.function.arguments
                                )
                            ) for tool_call in tool_calls
                        ]
                    )
                else:
                    content = Content(
                        text=message.content,
                    )

                if output_model and content.text != None:
                    try:
                        output_model.model_validate_json(content.text)
                        _messages.append(
                            Message(
                                role='assistant',
                                content=content,
                            )
                        )
                    except:
                        pass
                else:
                    _messages.append(
                        Message(
                            role='assistant',
                            content=content,
                        )
                    )
            
            if len(_messages) >= generation_config.n_candidates:
                _messages = _messages[:generation_config.n_candidates]
                break

            num_retry += 1

        if reduce_function:
            return reduce_function(_messages)
        else:
            return _messages

    async def a_generate(
        self,
        messages: List[Message],
        generation_config: GenerationConfig,
        reduce_function: Optional[Callable[[List[Message]], Message]]=None,
        output_model:BaseModel = None
    ) -> Union[Message, List[Message]]:
        # call the openai api

        kw_args = {key:value for key, value in generation_config.model_dump().items() if value != None and key in OEPNAI_API_KW}

        if output_model != None:
            messages.append(
                Message(
                    role='user', 
                    content=Content(
                        text='You must return a JSON object following this json schema. {schema}'.format(
                            schema=output_model.model_json_schema()
                        )
                    )
                )
            )
            kw_args['response_format'] = {'type':'json_object'}
        
        _messages = []

        num_retry = 0

        if generation_config.tools != None:
            kw_args['tools'] = [
                {
                    'type':'function',
                    'function':tool.model_dump()
                } for tool in generation_config.tools.values()
            ]
        
        if generation_config.api_type == 'azure':
            kw_args['model'] = generation_config.azure_deployment

        while num_retry < generation_config.max_retries:
            response = await self.a_client.chat.completions.create(
                messages=[transform_message_openai(message) for message in messages],
                **kw_args
            )
        
            generated_messages = [choice.message for choice in response.choices]

            for message in generated_messages:
                if message.tool_calls != None:
                    content = Content(
                        tool_calls=[
                            ToolCall(
                                id=tool_call.id,
                                type=tool_call.type, 
                                function_call=FunctionCall(
                                    name=tool_call.function.name.lower(), 
                                    arguments=tool_call.function.arguments
                                )
                            ) for tool_call in message.tool_calls
                        ]
                    )
                else:
                    content = Content(
                        text=message.content,
                    )

                if output_model and content.text != None:
                    try:
                        output_model.model_validate_json(content.text)
                        _messages.append(
                            Message(
                                role='assistant',
                                content=content,
                            )
                        )
                    except:
                        pass
                else:
                    _messages.append(
                        Message(
                            role='assistant',
                            content=content,
                        )
                    )
            
            if len(_messages) >= generation_config.n_candidates:
                _messages = _messages[:generation_config.n_candidates]
                break

            num_retry += 1
        
        if reduce_function:
            return reduce_function(_messages)
        else:
            return _messages