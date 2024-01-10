from ..beta import Beta
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field, field_validator
import openai

from ..beta import check_metadata,  generic_create, generic_retrieve, generic_update_metadata, generic_list_items


from typing import List, Dict, Optional, TypeVar, Type

from ..client import client

T = TypeVar('T', bound='MetaMessage')


class MetaMessage(Beta):

    id:Optional[str] = None
    object:Optional[str] = None
    created_at: Optional[int] = None
    metadata: Dict[str,str]=Field(default={})

    thread_id: str = Field(...)
    role:str = Field(...)
    content: List[Dict] = Field(...)

    assistant_id: Optional[str] = Field(default="")
    run_id: Optional[str] = Field(default="")
    
#    file_ids: Optional[List[str]] = Field(default=None)


    _check_metadata = field_validator('metadata')(classmethod(check_metadata))

    _do_not_include_at_creation_time = ['id', 'object', 'created_at', 'assistant_id', 'run_id']
    _do_not_include_at_update_time = ['id', 'object', 'created_at',  "role", "content",  "file_ids", 'assistant_id', 'run_id']
    
    @staticmethod
    def _create_fn(**kwargs):
        kwargs = {key:val for key, val in kwargs.items() if val != None or val != ""}
        content = kwargs['content'][0]['text']['value']
        
        kwargs['content'] = content
        if 'role' not in kwargs:
            kwargs['role'] = 'user'

        return client.beta.threads.messages.create(**kwargs)

    @staticmethod
    def _retrieve_fn(thread_id, message_id):
        return client.beta.threads.messages.retrieve(thread_id=thread_id, message_id=message_id)

    @staticmethod 
    def _custom_convert_for_retrieval(kwargs):
        kwargs['content'] = [{'type': x.type, 'annotations': getattr(x, x.type).annotations, 'value':getattr(x, x.type).value  } for x in kwargs['content']]
        return kwargs
        
    @staticmethod
    def _update_fn(thread_id, **kwargs):
        return client.beta.threads.messages.update(thread_id=thread_id, **kwargs)

    @staticmethod
    def _list_fn(**kwargs):
        return client.beta.threads.messages.list(**kwargs)
    

    @classmethod
    def create(cls:Type[T], **kwargs) -> T:
        content = kwargs['content']
        kwargs['content'] = [{'type': 'text', 'text': {'annotations': [], 'value': content}}]

        kwargs['message_type'] = cls.__name__
        
        cls._reference_class_abc = openai.types.beta.threads.thread_message.ThreadMessage 
        return generic_create(cls, **kwargs)
    
    @classmethod
    def retrieve(cls:Type[T], thread_id, message_id) -> T:
        cls._custom_convert_for_retrieval = cls._custom_convert_for_retrieval
        cls._reference_class_abc = openai.types.beta.threads.thread_message.ThreadMessage
        return generic_retrieve(cls, thread_id=thread_id, message_id=message_id)

    @classmethod
    def list(cls, **kwargs):
        cls._reference_class_abc = openai.types.beta.threads.thread_message.ThreadMessage
        return generic_list_items(cls, **kwargs)



    