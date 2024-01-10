from .metaassistant import MetaAssistant
from ..threads.basethread import BaseThread

from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Dict, List, Optional, TypeVar, Type
import re


import openai.types.beta
 

from ..beta import check_metadata,  generic_create, generic_retrieve, generic_update_metadata, generic_update, generic_delete, is_custom_field


from ..openai_functions import describe_all_openai_functions
from ..stream_thread import StreamThread
from ..session import Session
from ..client import client

T = TypeVar('T', bound='BaseAssistant')




class BaseAssistant (MetaAssistant):

    assistant_type:Optional[str] = Field(default="")

    session_threads_1: Optional[str] = Field(default="")
    session_threads_2: Optional[str] = Field(default="")
    session_threads_3: Optional[str] = Field(default="")
    session_threads_4: Optional[str] = Field(default="")

    pub_thread:Optional[str] = Field(default="")


    _assistant_thread_class = BaseThread
    
    def get_storage_attributes(self, list_type: str) -> List[str]:
        if list_type == "session":
            return ['session_threads_1', 'session_threads_2', 'session_threads_3', 'session_threads_4']
        else:
            return super().get_storage_attributes(list_type)
        
    def get_serde(self, list_type: str) :
        if list_type == "session":
            return Session
        else:
            return super().get_serde(list_type)



    @classmethod
    def create(cls:Type[T], **kwargs) -> T:
        kwargs['assistant_type'] = cls.__name__

        pub_thread = StreamThread.create()
        

        kwargs['pub_thread'] = pub_thread.id
        

        cls._reference_class_abc = openai.types.beta.assistant.Assistant 
        return generic_create(cls, **kwargs)
    

    @classmethod
    def retrieve(cls:Type[T], assistant_id) -> T:

        cls._custom_convert_for_retrieval = super()._custom_convert_for_retrieval
        cls._reference_class_abc = openai.types.beta.assistant.Assistant
        return generic_retrieve(cls, assistant_id=assistant_id)


    @classmethod
    def delete(cls:Type[T], assistant_id):
        _a = cls.retrieve(assistant_id=assistant_id)

        StreamThread.delete(thread_id=_a.pub_thread)
        
        _a._delete_all_sessions()

        cls._reference_class_abc = openai.types.beta.assistant.Assistant
        return generic_delete(cls=cls, assistant_id=assistant_id)


    def _delete_all_sessions (self):
        _load_sessions = self.load_list("session")
        for session in _load_sessions:
            
            main_thread_id = session.main_thread
            interaction_thread_id = session.interaction_thread
            print(main_thread_id)
            print(interaction_thread_id)

            self._assistant_thread_class.delete(main_thread_id)
            self._assistant_thread_class.delete(interaction_thread_id)

            

    def __init__(self, **data):
        
        if 'instructions' not in data:
            if self.__doc__ is not None:
                data['instructions'] = self.__doc__
            else:
                data['instructions'] = "NO INSTRUCTIONS IN CLASS DOC"

        super().__init__(**data)

        self._list_registry['session'] = self.__class__


    def _publish(self, content:str):

        pub_thread = StreamThread.retrieve(thread_id=self.pub_thread)
        pub_thread.publish_message(content=content)


    def get_sessions(self):
        return self.load_list("session")
    
    def update_session_attr_value(self, session_id, attribute, value):
        self.update_list_item_attr(list_type="session", list_id=session_id, list_item_attr=attribute, list_item_attr_value=value)
        

    def create_session(self, list_id:str):
        self.create_list_item("session", list_id=list_id)

    def delete_session (self, list_id:str):
        self.delete_list_item("session", list_id=list_id)


    def create_list_item(self, list_type,  list_id:str):
        if list_type == "session":
            sqlish_stmt = f"CREATE {list_type} IDENTIFIED BY {list_id}"
            print(sqlish_stmt)
            self._publish(content=sqlish_stmt)
        else: 
            super().create_list_item(list_type, list_id)


    def process_sqlish(self, sqlish:str):
        create_pattern = r"^CREATE"
        update_pattern = r"^UPDATE"
        delete_pattern = r"^DELETE"
        set_pattern = r"^SET"

        match = re.search(create_pattern, sqlish)
        if match:
            self._create_list_item(sqlish=sqlish)
            return "ok"
  
        match = re.search(update_pattern, sqlish)
        if match:
            self._update_list_item_attr(sqlish=sqlish)

        match = re.search(delete_pattern, sqlish)
        if match:
            self._delete_list_item(sqlish=sqlish)
        
        match = re.search(set_pattern, sqlish)
        if match: 
            self._set_attr(sqlish=sqlish)

    def _create_list_item (self, sqlish:str ) :

        pattern = r"CREATE (\w+) IDENTIFIED BY (.+)"
        match = re.search(pattern, sqlish)

        if match:
            list_type = match.group(1)
            if list_type == "session":
                list_id = match.group(2)
                _load_sessions = self.load_list(list_type=list_type)

                
                main_thread = self._assistant_thread_class.create()
                interaction_thread = self._assistant_thread_class.create()

                print(main_thread)
                print(interaction_thread)


                session = Session(assistant_id=self.id, session_id=list_id, 
                                main_thread=main_thread.id, 
                                interaction_thread=interaction_thread.id, 
                                session_type="d",
                                session_type_param = "d",  
                                session_model="" )

                _load_sessions.append(session)


                self.save_list('session', _load_sessions)
            else: 
                super()._create_list_item(sqlish)

        else: 
                print("no match check sqlish")
            


    def update_list_item_attr(self, list_type, list_id, list_item_attr, list_item_attr_value):

        if list_type == "session":
            sqlish_stmt = f"UPDATE {list_type} IDENTIFIED BY {list_id} SET {list_item_attr} = {list_item_attr_value}"
            print(sqlish_stmt)
            self._publish(content=sqlish_stmt)
        else: 
            super().create_list_item(list_type, list_id)


    def _update_list_item_attr(self, sqlish):
        pattern = r"UPDATE (\w+) IDENTIFIED BY ([\w-]+) SET ([\w-]+) = (.+)"
        match = re.search(pattern, sqlish)

        if match:
            list_type = match.group(1)
            if list_type == "session":
                list_id = match.group(2)
                field_name = match.group(3)
                field_value = match.group(4)

                _load_sessions = self.load_list(list_type=list_type)


                for (n, session)  in enumerate(_load_sessions):
                    if session.session_id == list_id:
                        session.__setattr__(field_name, field_value)
                        break                        

                self.save_list(list_type, _load_sessions)
            else: 
                super()._create_list_item(sqlish)

        else: 
                print("no match check sqlish")
            
        


    def delete_list_item(self,list_type:str, list_id:str):
        if list_type == "session":
            sqlish_stmt = f"DELETE {list_type} IDENTIFIED BY {list_id}"
            print(sqlish_stmt)
            self._publish(content=sqlish_stmt)
        else: 
            super().delete_list_item(list_type, list_id)


    def _delete_list_item (self, sqlish:str ) :

        pattern = r"DELETE (\w+) IDENTIFIED BY (.+)"
        match = re.search(pattern, sqlish)

        if match:
            list_type = match.group(1)
            if list_type == "session":
                list_id = match.group(2)
                _load_sessions = self.load_list(list_type=list_type)


                for (n, session)  in enumerate(_load_sessions):
                    if session.session_id == list_id:
                        self._assistant_thread_class.delete(session.main_thread)
                        self._assistant_thread_class.delete(session.interaction_thread)
                        del _load_sessions[n]
                        break                        

                self.save_list(list_type, _load_sessions)
            else: 
                super()._create_list_item(sqlish)

        else: 
                print("no match check sqlish")
            

    def set_attr(self, field_name, field_value) :
        content = f"SET {field_name} = {field_value}"
        
        self._publish(content=content)



    def _set_attr(self, sqlish:str):
        pattern = r"SET (\w+) = (.+)"
        match = re.search(pattern, sqlish)

        if match:
            field_name = match.group(1)
            field_value = match.group(2)

            self.__setattr__(field_name, field_value)

            if is_custom_field(self=self, field_name=field_name):
                generic_update_metadata(self=self)
            else:
                generic_update(self=self)
        else:
            print(f"error in SQLish")



         
class ClassedAssistant(BaseAssistant):
    assistant_class:str = Field(default=None)



class ProfessionalSentenceCreator(BaseAssistant):
    """You are a professional sentence creator. Using the sentence given to you, 
        please reword the sentence to make it sound more professional sounding 
        from the perspective of a senior product manager
    """
    pass

@describe_all_openai_functions()
class FunctionFinder(BaseAssistant):
    """ You are a function finder. You have access to a defined set of functions and the descriptions of the functions. 
        Your job is to figure out which subset of functions a specific person in a specific role will need. The set
         of functions is defined below with the name along with its description:

         # NAME : DESCRIPTION         

    """


    def __init__(self, **data):
        super().__init__(**data)
        
class StoryEditorAssistant(BaseAssistant): 
    """ You are a story editor assistant. """
    pass




if __name__ == "__main__":
#    tools = include_code_interpreter()
#    tools = include_retrieval(tools)
#    tools = include_all_openai_functions(tools)


#    ffa = FunctionFinder(name="functionFinder", tools=tools, model="gpt-3.5-turbo-1106")
#    ffa.create_assistant()

    sta = StoryEditorAssistant(name="storyEditorAssistant",  model="gpt-3.5-turbo-1106")
    sta.create_assistant()

    pta1 = ProfessionalSentenceCreator(name="lvl1_ProfessionalSentenceCreator", model="gpt-3.5-turbo-1106")
    pta2 = ProfessionalSentenceCreator(name="lvl2_ProfessionalSentenceCreator", model="gpt-4-1106-preview")


