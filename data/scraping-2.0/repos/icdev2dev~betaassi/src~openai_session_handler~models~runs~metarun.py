from typing import  Dict, Optional, Type, TypeVar
from pydantic import  Field, field_validator
import openai

from ..beta import Beta
from ..beta import check_metadata,  generic_create, generic_retrieve,  generic_list_items
from ..client import client

T = TypeVar('T', bound='MetaRun')

class MetaRun(Beta):
    id:Optional[str] = None
    object:Optional[str] = None
    created_at: Optional[int] = None
    metadata: Dict[str,str]=Field(default={})

    thread_id: str = Field(...)
    assistant_id:str = Field(...)
    
    status:Optional[str] = None
    required_action:Optional[str] = None
    last_error:Optional[Dict] = None
    expires_at:Optional[int] = None
    started_at:Optional[int] = None
    cancelled_at:Optional[int] = None
    failed_at:Optional[int] = None
    completed_at:Optional[int] = None
    model:Optional[str] = None
    instructions: Optional[str] = None

    # tools
    # file_ids
    #     
    _check_metadata = field_validator('metadata')(classmethod(check_metadata))
    _do_not_include_at_creation_time = ['id', 'object', 'created_at', 'status', 'started_at', 'expires_at', "cancelled_at", "failed_at", "completed_at", "last_error"]
    _do_not_include_at_update_time = ['id', 'object', 'created_at', 'status', 'started_at', 'expires_at', "cancelled_at", "failed_at", "completed_at", "last_error", "model", "instructions", "tools", "file_ids"]

    @staticmethod
    def _create_fn(**kwargs):
        kwargs = {key:val for key, val in kwargs.items() if val != None}
        return client.beta.threads.runs.create(**kwargs)

    @staticmethod
    def _retrieve_fn(thread_id, run_id):
        return client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)    

    @staticmethod 
    def _custom_convert_for_retrieval(kwargs):
        return kwargs

    @staticmethod
    def _update_fn(thread_id, run_id, **kwargs):
        return client.beta.threads.runs.update(thread_id=thread_id, run_id=run_id, **kwargs)
    
    @staticmethod
    def _list_fn(thread_id):
        return client.beta.threads.runs.list(thread_id=thread_id)


    @classmethod
    def create(cls:Type[T], **kwargs) -> T:
        kwargs['run_type'] = cls.__name__

        cls._reference_class_abc = openai.types.beta.threads.run.Run 
        return generic_create(cls, **kwargs)


    @classmethod
    def retrieve(cls:Type[T], thread_id, run_id) -> T:

        cls._reference_class_abc = openai.types.beta.threads.run.Run
        return generic_retrieve(cls, thread_id=thread_id, run_id=run_id)

    

    @classmethod
    def list(cls, **kwargs):
        cls._reference_class_abc = openai.types.beta.threads.run.Run
        return generic_list_items(cls, **kwargs)
    

class BaseRun(MetaRun):
    run_type:Optional[str] = Field(default="")

