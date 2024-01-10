from ..beta import Beta
from typing import Optional, Dict,  Type, TypeVar
from pydantic import Field, field_validator

from ..beta import check_metadata, generic_create,generic_delete, generic_retrieve, generic_update, generic_update_metadata

import openai
from ..client import client

#from base_message import BaseMessage
#from base_run import BaseRun



T = TypeVar('T', bound='MetaThread')


class MetaThread (Beta):
        # ALL fields from openai threads

    id:Optional[str] = None
    object:Optional[str] = None
    created_at: Optional[int] = None
    metadata: Dict[str,str]=Field(default={})

    _check_metadata = field_validator('metadata')(classmethod(check_metadata))
    _do_not_include_at_creation_time = ['id', 'object', 'created_at', 'assistant_id', 'run_id']
    _do_not_include_at_update_time = ['id', 'object', 'created_at']
    


    @staticmethod
    def _create_fn(**kwargs):
        return client.beta.threads.create(**kwargs)

    @staticmethod
    def _retrieve_fn(thread_id):
        return client.beta.threads.retrieve(thread_id=thread_id)

    @staticmethod 
    def _custom_convert_for_retrieval(kwargs):
        return kwargs

    @staticmethod
    def _update_fn(thread_id, **kwargs):
        return client.beta.threads.update(thread_id=thread_id, **kwargs)

    @staticmethod
    def _delete_fn(thread_id, **kwargs):
        return client.beta.threads.delete(thread_id=thread_id)
    
        
    @classmethod
    def create(cls:Type[T], **kwargs) -> T:
        cls._reference_class_abc = openai.types.beta.thread.Thread 
        kwargs['thread_type'] = cls.__name__

        return generic_create(cls, **kwargs)
    
    @classmethod
    def retrieve(cls:Type[T], thread_id) -> T:

        cls._custom_convert_for_retrieval = cls._custom_convert_for_retrieval
        cls._reference_class_abc = openai.types.beta.thread.Thread
        return generic_retrieve(cls, thread_id=thread_id)

    @classmethod
    def delete(cls:Type[T], thread_id) :
        cls._reference_class_abc = openai.types.beta.thread.Thread
        return generic_delete(cls=cls, thread_id=thread_id)





        
