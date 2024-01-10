import requests
from utils import (
    convert_messages,
    query_ai_user_dict,
    create_assistant_base_pompt,
    create_vector_store_context_prompt,
    create_web_context_prompt,
    create_web_query_prompt,
    emit_socket_event
)
from embedding_service import embedding_service
from qdrant_vector_store import qdrant_vector_store
from langchain_plugin.plugin_executor import PluginExecutor, get_plugin, GOOGLE_SEARCH
import openai
from map_reduce_text_summarizer import MapReduceTextSummarizer
import time
from chatbot import ChatBot

class ChatRoomAnswerService:
    ai_user_dict = query_ai_user_dict()
    CHATROOM_SERVER = "http://chatroom-server:5000"
    def __init__(self, prompt: str, room_rule: str, api_key: str, room_id: str, user_id: str, user_name: str,messages: list[dict[str, str]]) -> None:
        self.prompt = prompt
        self.room_rule = room_rule
        self.__validate_prompt(self.prompt, 3000)
        openai.api_key = api_key
        self.room_id = room_id
        self.user_id = user_id
        self.user_name = user_name
        self.messages = messages
        self.messages_with_prompt = convert_messages([*messages, {"user_id": self.user_id, "content": self.prompt}])
        self.ai_id = self.ai_user_dict["user_id"]
        self.ai_name = self.ai_user_dict["user_name"]
    def ask_vector_store(self) -> str:
        querys = [
            message["content"] 
            for message in self.messages_with_prompt 
            if message["role"] == "user"
        ]
        print(f"Querying vector store with query: {querys}", flush= True)
        all_relevant_docs = self.__get_relevant_docs("\n".join(querys))
        all_logs = "\n".join([f"{str(doc)}" for doc in all_relevant_docs])
        all_logs = self.__summarize_if_text_exceed_context_window_limit(all_logs, 8000)
        system_prompt = create_assistant_base_pompt(self.room_rule) + create_vector_store_context_prompt(all_logs)
        self.__emit_message_to_room(self.user_id,self.user_name, self.prompt)
        bot = self.__bot_answer(system_prompt)
        return  {
            **bot.bot_response.to_dict(),
            "user_id": self.user_id,
            "room_id": self.room_id,
            "sources": list(set([
                doc.get("document_id") for doc in all_relevant_docs
            ]))
        }
    
    def __get_relevant_docs(self, query: str) -> list[dict[str, str]]:
        embedding = embedding_service.get_embedding(query, None, None)
        query_results = qdrant_vector_store.search_text_chunks(self.room_id, embedding, threshold= 0.8)
        relevant_docs = []
        for payload in query_results:
            document_id = payload["document_id"]
            chunk_id = payload["chunk_id"]
            contexts = embedding_service.get_adjancent_embeddings(document_id, chunk_id)
            relevant_docs.append(payload)
            relevant_docs.extend(contexts)
        return relevant_docs

    
    def ask_web(self) -> str:
        '''
        Asking web requires langchain plugin
        '''
        google_search_plugin = get_plugin(GOOGLE_SEARCH)
        executor = PluginExecutor(google_search_plugin)
        plugin_param = {
            "-o": openai.api_key,
            "-s": "090d6a2fac0a459e894a5b1aa17674b0def6ff34",
        }
        
        self.__emit_message_to_room(self.user_id,self.user_name, self.prompt)

        query_prompt = create_web_query_prompt(self.prompt, self.messages)
        for log in executor.execute(query_prompt, plugin_param):
            all_log = "\n".join(log)
            emit_socket_event(f"thinking/{self.room_id}", all_log)
        all_log = executor.get_logs()
        
        all_log = self.__summarize_if_text_exceed_context_window_limit(all_log, 8000)
            

        system_prompt = create_assistant_base_pompt(self.room_rule) + create_web_context_prompt(all_log)
        bot = self.__bot_answer(system_prompt)
        
        return  {
            **bot.bot_response.to_dict(),
            "user_id": self.user_id,
            "room_id": self.room_id,
            "sorces": all_log
        }
    
    def __summarize_if_text_exceed_context_window_limit(self, text: str, context_window_limit: int = 8000) -> str:
        if len(text) < context_window_limit:
            print(f"Content length is less than {context_window_limit} words, getting {len(text)}, no need to summarize", flush= True)
            return text
        print(f"Content truncated warning: logs exceed {context_window_limit} words: getting {len(text)}, summarized into 4000 words article", flush= True)
        summarizer = MapReduceTextSummarizer(openai_api_key= openai.api_key, question= self.prompt, text= text)
        return summarizer.summarize()
    
    def __bot_answer(self, system_prompt: str) -> ChatBot:
        print(system_prompt, flush= True)
        
        bot = ChatBot(system_prompt, self.messages_with_prompt)
        for current_message in bot.answer():
            self.__emit_message_to_room(self.ai_id ,self.ai_name,current_message)
        
        self.__emit_message_to_room(self.user_id,self.user_name, self.prompt, is_message_persist= True, is_emit= False)
        time.sleep(1)
        self.__emit_message_to_room(self.ai_id ,self.ai_name,bot.current_message, is_message_persist= True)
        print(flush= True)
        return bot

    def __emit_message_to_room(self, user_id: str, user_name: str,content:str, is_message_persist: bool = False, is_emit: bool = True) -> requests.Response | None:
        '''
        {
            "message_type": "regular" | "ai"
            "room_id": str
            "user_id": str
            "content": str
            "is_message_persist": bool
        } 
        '''
        message_type = "ai"

        post_json = {
            "message_type": message_type,
            "room_id": self.room_id,
            "user": {
                "user_id": user_id,
                "user_name": user_name,
            },
            "content": content,
            "is_message_persist": is_message_persist,
            "is_ai": True,
            "is_emit": is_emit,
        }
        if is_message_persist: # only post if true
            response = requests.post(f"{self.CHATROOM_SERVER}/emit_message_to_room", json= post_json)
            return response.json()
        
        msg = {"user_id": user_id, "user_name": user_name,"content": content, "room_id": self.room_id, "is_message_persist": is_message_persist}
        emit_socket_event(f"{message_type}/{self.room_id}", msg)
    
    def __validate_prompt(self, prompt: str, size_limit: int = 15000) -> None:
        prompt_length = len(prompt)

        if prompt_length == 0:
            raise ValueError(f"Abort empty text memo request")

        if prompt_length > size_limit:
            raise ValueError(f"Prompt length must shorter than {size_limit}, entering {prompt_length}")