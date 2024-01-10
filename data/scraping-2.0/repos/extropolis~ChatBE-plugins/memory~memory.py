import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable

import dotenv
import numpy as np
import openai
import pinecone
from google.cloud.firestore_v1.base_client import BaseClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ..base import BaseTool
from .CustomSTEmbeddings import CustomSTEmbeddings

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_KEY"]
os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"] # for langchain compatibility

DEFAULT_USER_FACT = {
    "name": None,
    "age": None,
    "gender": None,
    "home_city": None,
    "occupation": None,
    "employer": None,
    "spouce": None,
    "friends": [],
    "kids": [],
    "interests": [],
}


class Memory:
    def __init__(self, db: BaseClient=None, short_memory_length: int=10, 
                 model="gpt-3.5-turbo-16k", embedding_type="gpt", history_limit=3):
        '''
            short_memory_length: length of conversation considered for the short memory. This is the number of user messages. if short_memory_length=1, the short memory will actually include at least 3 messages (1 initial message from AI, 1 from user, 1 from the AI assistant.)

            embedding_type can be chosen from "gpt" for OpenAI embedding or "st" sentence transformer embedding,
            For now, st may be incompatible, due to different embedding size.

            history_limit: what's the maximum number of answers for a query to consider for repetitiveness
        '''
        if db == None:
            raise Exception("database is None when initializing memory")
        self.memory = db.collection("memory")
        self.active_user_sessions = defaultdict(list)
        self.user_short_memory = defaultdict(list)
        self.short_mem_len = short_memory_length
        self.model = model
        if self.model.endswith("16k"):
            self.threshold = 14000
        elif self.model.endswith("32k"):
            self.threshold = 28000
        else:
            self.threshold = 2000

        self.history_limit = history_limit

        self.embedding_type = embedding_type

        index_name = "fileidx"
        dimension = 1536
        if embedding_type == "gpt":
            embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])
            self.similarity_thres = 0.85
            self.retrieve_function = self.retrieve_interactions_from_similar_queries_vector_store
            pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_API_ENV"])
            self.pinecone_db = Pinecone(index=pinecone.Index(index_name), embedding=embeddings, text_key="text")
        elif embedding_type == "st":
            try:
                import torch
                model_kwargs = {"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
            except:
                model_kwargs = {"device": "cpu"}
            embeddings = CustomSTEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2", 
                model_kwargs=model_kwargs,
                embedding_dimension=dimension)
            self.similarity_thres = 0.75
            self.retrieve_function = self.retrieve_interactions_from_similar_queries_vector_store
            pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_API_ENV"])
            self.pinecone_db = Pinecone(index=pinecone.Index(index_name), embedding=embeddings, text_key="text")
        else:
            self.similarity_thres = 0.9
            self.retrieve_function = self.retrieve_interactions_from_similar_queries
            
        # uncomment these for real production, slow check
        # try:
        #     if index_name not in pinecone.list_indexes():
        #         print("Creating pinecone index")
        #         pinecone.create_index(index_name, dimension)
        # except Exception as e:
        #     print(e)

    def store_user_facts(self, user_id, facts):
        user_fact_ref = self.memory.document(user_id)
        user_fact_doc = user_fact_ref.get()
        if user_fact_doc and user_fact_doc.exists:
            new_user_facts = user_fact_doc.to_dict()
            for k, v in facts.items():
                if v is not None and isinstance(v, (str, int, dict)):
                    new_user_facts[k] = v
                elif isinstance(v, list):
                    new_user_facts[k].extend(v)
                    # remove duplicates
                    new_user_facts[k] = list(set(new_user_facts[k]))
            user_fact_ref.update(new_user_facts)
            print(f"GPT generated facts:\n {json.dumps(facts, indent=True)}")
            print(f"Updated user facts:\n {json.dumps(new_user_facts, indent=True)}")
        else:
            user_fact_ref.set(facts)
            print(facts)
    
    def get_memory(self, user_id):
        user_fact_ref = self.memory.document(user_id)
        user_fact_doc = user_fact_ref.get()
        if user_fact_doc and user_fact_doc.exists:
            return user_fact_doc.to_dict()
        else:
            facts = DEFAULT_USER_FACT.copy()
            return facts
    
    def __trim_conversation(self, conversations):
        total_len = 0
        for i in range(len(conversations) - 1, -1, -1):
            total_len += len(conversations[i]["content"].split(" ")) + 1 # to also account for the role token
            if total_len > self.threshold:
                print(f"Session length: {total_len}")
                return conversations[i:]
        print(f"Session length: {total_len}")
        return conversations
    
    async def generate_user_facts(self, user_id):
        user_facts = self.get_memory(user_id)
        new_message = """Based on the conversation history, update the following information about the 'user' in JSON format. Output only the completed JSON object. Write 'null' if the information does not exist. You should focus on what the user says. The assistant messages should only be taken as references and not factual about the user.
        Format:
        {
            "name": <string>,
            "age": <int>,
            "gender": <string>,
            "home_city": <string>,
            "occupation": <string>,
            "employer": <string>,
            "spouce": <string>,
            "friends": [],
            "kids": [],
            "interests": [],
        }"""
        new_message += f"""
        Known user facts:
        {json.dumps(user_facts, indent=True)}
        """

        # This is just for safety, in case we have too large a short_mem_len set
        short_conversation = self.__trim_conversation(self.user_short_memory[user_id])

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages= short_conversation + [{"role": "user", "content": new_message}],
            temperature=0
        )
        
        try:
            content = response['choices'][0]['message']['content']
            user_facts = json.loads(content)
            # user_facts["last_conversation"] = date.today().isoformat()
        except Exception as e:
            print(e)

        self.store_user_facts(user_id, user_facts)

    async def generate_follow_up(self, user_id):
        print("GENERATING FOLLOW UPS")
        user_facts = self.get_memory(user_id)
        new_message = """Summarize the conversation and generate some follow up ideas to start the next conversation. Output only the complete JSON object. The summarization and each follow up idea should not exceed 100 words.
        Format:
        {
            "summary": <string>,
            "follow_up": [],
        }"""
        user_msg_count = sum(1 for d in self.active_user_sessions[user_id] if d["role"] == "user")
        if user_msg_count < 1:
            print("No user interaction in the conversation, should not overwrite the conversation history")
            return

        conversations = self.__trim_conversation(self.active_user_sessions[user_id])

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=conversations + [{"role": "user", "content": new_message}],
            # temperature=0
        )

        try:
            content = response['choices'][0]['message']['content']
            print(content)
            conversation_summary = json.loads(content)
            conversation_summary["timestamp"] = datetime.now().isoformat()
            print(conversation_summary)
            user_facts["last_conversation"] = conversation_summary
            # user_facts["last_conversation"] = date.today().isoformat()
        except Exception as e:
            print(e)
        self.store_user_facts(user_id, user_facts)

    async def update_memory(self, user_id, user_session_finished=False):
        if len(self.active_user_sessions[user_id]) == 0:
            print("nothing to update")
            return
        user_msg_count = sum(1 for d in self.user_short_memory[user_id] if d["role"] == "user")
        if user_msg_count < len(self.user_short_memory[user_id]) // 2 or len(self.user_short_memory[user_id]) == 0:
            # user has talked too little, should not update the memory
            print("User has talked too little, do not update short term memory")
        else:
            await self.generate_user_facts(user_id)
        self.user_short_memory[user_id].clear()

        if user_session_finished:
            await self.generate_follow_up(user_id)
            self.active_user_sessions[user_id].clear()
    
    def clear_memory(self, user_id):
        user_fact_ref = self.memory.document(user_id)
        user_fact_doc = user_fact_ref.get()
        if user_fact_doc and user_fact_doc.exists:
            user_fact_ref.delete()
        else:
            return

    def update_user_session(self, user_id, message):
        '''Only insert the message to the memory for simplicity. If we want to do any other processing, use other functions.'''
        assert isinstance(message, dict) and "role" in message and "content" in message, "message must be a dictionary containing role and content fields"
        assert message["role"] in ["user", "assistant"], "message role must be one of user, assistant"
        self.active_user_sessions[user_id].append(message)
        # self.user_short_memory[user_id].append(message)

        # user_msg_count = sum(1 for d in self.user_short_memory[user_id] if d["role"] == "user")
        # if user_msg_count >= self.short_mem_len and self.user_short_memory[user_id][-1]["role"] == "assistant":
        #     # Hacky way of checking the messages, need a better approach
        #     await self.update_memory(user_id)
        #     print("memory about user updated")
    
    def restrict_tokens(self, closest_interactions):
        i = 0
        token_counts = 0
        while i < len(closest_interactions):
            token_counts += len(closest_interactions[i][0].split(" ")) + len(closest_interactions[i][1].split(" ")) + 2
            if token_counts > 200:
                # in case we go over token limit, keep only around 200 tokens
                break
            i += 1
        closest_interactions = closest_interactions[:i+1]

        # construct a proper history for GPT to understand
        hist = []
        for i in range(len(closest_interactions)):
            hist.append({"role": "user", "content": closest_interactions[i][0]})
            hist.append({"role": "assistant", "content": closest_interactions[i][1]})
        return hist

    def retrieve_interactions_from_similar_queries(self, user_id, user_assistants):
        '''For now, we simply use a tf-idf vectorizer and knn search'''
        if user_assistants is None or len(user_assistants) == 0:
            return
        user_current_session = self.active_user_sessions[user_id]
        assert user_current_session[-1]["role"] == "user"
        user_msgs = [(msg["content"], user_current_session[idx+1]["content"]) \
                     for idx, msg in enumerate(user_current_session) \
                     if msg["role"] == "user" and idx < len(user_current_session) - 1]
        user_msgs.append((user_current_session[-1]["content"], ""))

        start_time = time.time()
        vectorizer = TfidfVectorizer(encoding="utf-8", decode_error="ignore", stop_words="english")
        X = vectorizer.fit_transform([m[0] for m in user_msgs]).toarray()
        print(f"Shape: {X.shape}, Time taken to vectorize all the data: {time.time() - start_time}")

        start_time = time.time()
        nn_finder = NearestNeighbors(n_neighbors=5)
        nn_finder.fit(X)
        print(f"Time taken to find nn: {time.time() - start_time}")

        dist, neigh = nn_finder.kneighbors([X[-1]], min(5, len(user_msgs)))
        dist = dist.reshape(-1)
        neigh = neigh.reshape(-1)
        # the values should already be sorted in increasing order of distance
        closest_interactions = []
        for i in range(len(dist)):
            if neigh[i] == len(user_msgs) - 1:
                # do not use the latest message
                continue
            interaction = user_msgs[neigh[i]]
            print(f"With sentence: {interaction[0]}, distance: {dist[i]}")
            if 0 <= dist[i] < self.similarity_thres:
                closest_interactions.append(interaction)
            if len(closest_interactions) > self.history_limit:
                # retrieve at most self.history_limit interactions
                break
        
        if len(closest_interactions) == 0:
            return
        
        hist = self.restrict_tokens(closest_interactions)
        # inject to the bots
        info = f"Do not repeat yourself with the following interactions: {json.dumps(hist)}"
        for assistant in user_assistants:
            assistant(info)

    def insert_vector_store(self, user_id):
        '''This should be used when the response from GPT is finished'''
        user_current_session = self.active_user_sessions[user_id]
        assert len(user_current_session) >= 2
        assert user_current_session[-1]["role"] == "assistant"
        assert user_current_session[-2]["role"] == "user"
        query = user_current_session[-2]["content"]
        answer = user_current_session[-1]["content"]
        id = query.replace(" ", "-")
        metadata = {
            "answers": [answer]
        }

        # check if the same query exists already
        # if exists, append the answer to the previous id
        # otherwise, insert the query
        start_time = time.time()
        search_results = self.pinecone_db.similarity_search_with_score(query, k=self.history_limit, namespace=f"{user_id}-history")
        for (doc, score) in search_results:
            if doc.page_content == query:
                doc.metadata["answers"].append(answer)
                while len(doc.metadata["answers"]) > self.history_limit:
                    doc.metadata["answers"].pop(0)
                self.pinecone_db.delete(ids=[id], namespace=f"{user_id}-history")
                metadata = doc.metadata
                break
        self.pinecone_db.add_texts(texts=[query], metadatas=[metadata], ids=[id], namespace=f"{user_id}-history")
        print(f"Time to search similar and insert query: {query} :: {time.time() - start_time}")

    def retrieve_interactions_from_similar_queries_vector_store(self, user_id, user_assistants):
        '''In this function, we use Pinecone vector store, with either sentence transformer or OpenAI embeddings'''
        user_current_session = self.active_user_sessions[user_id]
        assert len(user_current_session) >= 1
        assert user_current_session[-1]["role"] == "user"
        query = user_current_session[-1]["content"]
        start_time = time.time()
        search_results = self.pinecone_db.similarity_search_with_score(query, k=self.history_limit, namespace=f"{user_id}-history")

        closest_interactions = []
        for (doc, score) in search_results:
            if score < self.similarity_thres:
                continue
            for ans in doc.metadata["answers"]:
                if len(closest_interactions) < self.history_limit:
                    closest_interactions.append((query, ans))
        print(f"Time to search similar: {time.time() - start_time}, similar queries: {closest_interactions}")

        if len(closest_interactions) == 0:
            return
        
        hist = self.restrict_tokens(closest_interactions)
        # inject to the bots
        info = f"Do not repeat yourself with the following interactions: {json.dumps(hist)}"
        for assistant in user_assistants:
            assistant(info)
        
class MemoryTool(BaseTool):
    name: str = "memory"
    description: str = "Tool for memorizing facts about users. Contains long term and short term memory about users. When this tool is enabled, initial prompts for AI assistants will be updated according to known user facts. However, this tool itself should not be used by the AI assistants."
    user_description: str = "You can enable this to get a better continuation between different conversations."
    usable_by_bot: bool = False
    def __init__(self, func: Callable=None, **kwargs) -> None:
        db = kwargs.get("db", None)
        short_memory_length = kwargs.get("short_memory_length", 5)
        memory_model = kwargs.get("memory_model", "gpt-3.5-turbo-16k")
        memory_embedding_type = kwargs.get("memory_embedding_type", "gpt")
        memory_history_limit = kwargs.get("memory_history_limit", 3)
        self.memory = Memory(db=db, short_memory_length=short_memory_length, model=memory_model,
                             embedding_type=memory_embedding_type,
                             history_limit=memory_history_limit)
        
        self.repetitive_check = kwargs.get("memory_repetitive_check", False)
        # All the handlers must have been correctly setup, otherwise Memory is no use, 
        # so if there is any error, we must raise
        OnStartUp = kwargs.get("OnStartUp")
        OnStartUpMsgEnd = kwargs.get("OnStartUpMsgEnd")
        OnUserMsgReceived = kwargs.get("OnUserMsgReceived")
        OnResponseEnd = kwargs.get("OnResponseEnd")
        OnUserDisconnected = kwargs.get("OnUserDisconnected")
        OnStartUp += self.OnStartUp
        OnStartUpMsgEnd += self.OnStartUpMsgEnd
        OnUserMsgReceived += self.OnUserMsgReceived
        OnResponseEnd += self.OnResponseEnd
        OnUserDisconnected += self.OnUserDisconnected

        super().__init__(None)

    def OnStartUp(self, **kwargs):
        user_tool_settings = kwargs.get("user_tool_settings", {self.name: False})
        user_id = kwargs.get("user_id")
        user_info = kwargs.get("user_info")
        if user_tool_settings[self.name]:
            mem = self.get_memory(user_id)
            for k, v in mem.items():
                if k not in ["name", "interests"]:
                    user_info[k] = v
    
    def OnStartUpMsgEnd(self, **kwargs):
        user_tool_settings = kwargs.get("user_tool_settings", {self.name: False})
        user_id = kwargs.get("user_id")
        user_info = kwargs.get("user_info")
        message = kwargs.get("message")
        if user_tool_settings[self.name]:
            self.memory.update_user_session(user_id, message)

    def OnUserMsgReceived(self, **kwargs):
        user_tool_settings = kwargs.get("user_tool_settings", {self.name: False})
        user_id = kwargs.get("user_id")
        user_info = kwargs.get("user_info")
        message = kwargs.get("message")
        user_assistants = kwargs.get("user_assistants", [])
        if user_tool_settings[self.name]:
            self.memory.update_user_session(user_id, message)
            if self.repetitive_check:
                self.memory.retrieve_function(user_id, user_assistants)
            # self.memory.retrieve_interactions_from_similar_queries_vector_store(user_id, user_assistants)
            # mem = self.get_memory(user_id)
            # for k, v in mem.items():
            #     user_info[k] = v

    def OnResponseEnd(self, **kwargs):
        user_tool_settings = kwargs.get("user_tool_settings", {self.name: False})
        user_id = kwargs.get("user_id")
        user_info = kwargs.get("user_info")
        message = kwargs.get("message")
        if user_tool_settings[self.name]:
            self.memory.update_user_session(user_id, message)
            if self.repetitive_check and self.memory.embedding_type in ["gpt", "st"]:
                self.memory.insert_vector_store(user_id)

    def OnUserDisconnected(self, **kwargs):
        user_tool_settings = kwargs.get("user_tool_settings", {self.name: False})
        user_id = kwargs.get("user_id")
        if user_tool_settings[self.name]:
            asyncio.create_task(self.update_memory(user_id, True))
    
    def get_memory(self, user_id):
        return self.memory.get_memory(user_id)

    async def update_memory(self, user_id, user_session_finished=False):
        await self.memory.update_memory(user_id, user_session_finished)
    
    async def clear_memory(self, user_id):
        self.memory.clear_memory(user_id)

    async def update_user_session(self, user_id, message):
        await self.memory.update_user_session(user_id, message)

    def on_enable(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def on_disable(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def _run(self, query: str):
        return None

