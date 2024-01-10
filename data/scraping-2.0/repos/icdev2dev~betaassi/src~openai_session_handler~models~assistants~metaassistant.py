
from ..beta import Beta
from pydantic import Field, field_validator
from ..session import Session
from typing import List, Optional, Dict, Type, TypeVar
from ..client import client

from ..beta import check_metadata,  generic_create, generic_retrieve, generic_update_metadata, generic_update, generic_delete, is_custom_field
import openai



T = TypeVar('T', bound='MetaAssistant')


class MetaAssistant(Beta):
    
    # ALL FIELDS FROM OPENAI ASSISTANT
    id:Optional[str] = None
    object:Optional[str] = None
    created_at: Optional[int] = None
    metadata: Dict[str,str]=Field(default={})


    name: str = Field(default="ABC", max_length=512)
    description: str = Field(default="Default Description", max_length=512)
    instructions: str = Field(...)
    
    model:str = Field(default="gpt-4", max_length=512)    
    
#    tools: Optional[List[Dict]] = Field(default=None)
#    file_ids: Optional[List[str]] = Field(default=None)



    _check_metadata = field_validator('metadata')(classmethod(check_metadata))
    _do_not_include_at_creation_time = ['id', 'object', 'created_at']
    _do_not_include_at_update_time = ['id', 'object', 'created_at']


    # ALL OPEN AI CRUDL Functions MUST BE DEFINED HERE

    @staticmethod
    def _create_fn(**kwargs):
        return client.beta.assistants.create(**kwargs)

    @staticmethod
    def _retrieve_fn(assistant_id):
        return client.beta.assistants.retrieve(assistant_id=assistant_id)
    
    @staticmethod  
    def _custom_convert_for_retrieval(kwargs):
        return kwargs

    @staticmethod
    def _update_fn(assistant_id, **kwargs):
        return client.beta.assistants.update(assistant_id=assistant_id, **kwargs)

    @staticmethod
    def _delete_fn(assistant_id):
        return client.beta.assistants.delete(assistant_id=assistant_id)
    
    @staticmethod
    def _list_fn(**kwargs):
        return client.beta.assistants.list(**kwargs)



    @classmethod
    def create(cls:Type[T], **kwargs) -> T:
        cls._reference_class_abc = openai.types.beta.assistant.Assistant 
        return generic_create(cls, **kwargs)
    

    @classmethod
    def retrieve(cls:Type[T], assistant_id) -> T:

        cls._custom_convert_for_retrieval = cls._custom_convert_for_retrieval
        cls._reference_class_abc = openai.types.beta.assistant.Assistant
        return generic_retrieve(cls, assistant_id=assistant_id)


    @classmethod
    def delete(cls:Type[T], assistant_id):
        cls._reference_class_abc = openai.types.beta.assistant.Assistant
        return generic_delete(cls=cls, assistant_id=assistant_id)


























class MetaAssistant2(Beta):
    f1:str = Field(default="")
    f2:str = Field(default="")
    f3:str = Field(default="")

    def get_storage_attributes(self, list_type: str) -> List[str]:
        if list_type == "session":
            return ['f1', 'f2', 'f3']
        else:
            return super().get_storage_attributes(list_type)
        
    def get_serde(self, list_type: str) :
        if list_type == "session":
            return Session
        else:
            return super().get_storage_attributes(list_type)




    def __init__(self, **data):
        super().__init__(**data)
        self._list_registry['session'] = MetaAssistant
