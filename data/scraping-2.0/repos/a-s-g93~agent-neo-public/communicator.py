from neo4j.exceptions import ConstraintError
from google.oauth2 import service_account
from google.cloud import aiplatform
import streamlit as st
import drivers
from typing import List, Optional
from graphdatascience import GraphDataScience
from credentials import neo4j_credentials

from vertexai.preview.language_models import TextEmbeddingModel

from langchain.chat_models import ChatVertexAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

import time
import uuid

import openai
from langchain.chat_models import AzureChatOpenAI

from ratelimit import limits, RateLimitException, sleep_and_retry

EMBEDDING_DIMENSIONS = 768
TEXT_EMBEDDING_MODEL = "textembedding-gecko@001"
TEXT_GEN_MODEL = "text-bison@001"
CHAT_MODEL = 'chat-bison@001'
CHAT_MODEL_32 = 'chat-bison-32k'


# AUTHENTICATE SERVICE ACCOUNT  
google_credentials = service_account.Credentials.from_service_account_info(
    st.secrets["google_service_account"]
)

# AUTHENTICATE OPENAI
openai.api_key = st.secrets['openai_key']
openai.api_version = st.secrets['openai_version']

class Communicator:
    """
    The constructor expects an instance of the Neo4j Driver, which will be
    used to interact with Neo4j.
    This class contains methods necessary to interact with the Neo4j database
    and manage conversations with the chosen LLM.
    """

    def __init__(self) -> None:

        self.driver = drivers.init_driver(neo4j_credentials['uri'], username=neo4j_credentials['username'], password=neo4j_credentials['password'])
        self.database_name = neo4j_credentials['database']
        self.project = st.secrets['gcp_project']
        self.region = st.secrets['gcp_region']

        # init the aiplatform 
        aiplatform.init(project=self.project, location=self.region, credentials=google_credentials)
        # vertexai.init(project=self.project, location=self.region, credentials=google_credentials)

        # instantiate Google text embedding model
        self.text_embedding_model = TextEmbeddingModel.from_pretrained(TEXT_EMBEDDING_MODEL)

        # init Neo4j GDS client
        self.gds = GraphDataScience(neo4j_credentials['uri'], auth=(neo4j_credentials['username'], neo4j_credentials['password']), aura_ds=False)

    def test_database_connection(self):
        """
            This function tests database connection.
        """

        def test(tx):
            return tx.run(
                """
                match (d:Document)
                return d
                limit 1
                """
            ).data()
   
        try:
            with self.driver.session(database=self.database_name) as session:
                print("database name: ", self.database_name)
                result = session.execute_read(test)
                print("result: ", result)
            
                return result
            
        except ConstraintError as err:
            print(err)

            session.close()
    
    # 25 calls / 20 hours
    @limits(calls=25, period=72000)
    def encode_texts_to_embeddings(self, sentences: List[str]) -> List[Optional[List[float]]]:
        try:
            embeddings = self.text_embedding_model.get_embeddings(sentences)
            return [embedding.values for embedding in embeddings]
        except Exception:
            return [None for _ in range(len(sentences))]
        
    def create_context(self, question):
        '''
        This function takes the user question and creates an embedding of it
        using a vertexai model. 
        Cosine similarity is ran on the embedding against the embeddings in the 
        Neo4j database to find documents that will be used to construct
        the context. 
        The top n documents with their URLs are returned as context.
        '''

        # generate embeddings from matching prompt
        embeddings = self.encode_texts_to_embeddings(sentences=[question])
        st.session_state['recent_question_embedding'] = embeddings[0]
        
        def neo4j_vector_index_search():
            """
            This method runs vector similarity search on the document embeddings against the question embedding.
            """

            return self.gds.run_cypher("""
                                        CALL db.index.vector.queryNodes('document-embeddings', toInteger($k), $questionEmbedding)
                                        YIELD node AS vDocs, score
                                        return vDocs.url as url, vDocs.text as text, vDocs.index as index
                                       """, {'questionEmbedding': embeddings[0], 'k': st.session_state['num_documents_for_context']}
                                       )
        
        if st.session_state['num_documents_for_context'] > 0:
            # get documents from Neo4j database
            neo4j_timer_start = time.perf_counter()
            try:
                # docs = neo4j_cosine_similarity()
                docs = neo4j_vector_index_search()
            except ConstraintError as err:
                print(err)
                
            print("Neo4j time: "+str(round(time.perf_counter()-neo4j_timer_start, 4))+" seconds.")
        else:
            print('no context retrieved...')
            docs = None

        return docs
    
    def log_new_conversation(self, llm, user_input):
        """
        This method creates a new conversation node and logs the 
        initial user message in the neo4j database.
        Appropriate relationships are created.
        """

        log_timer_start = time.perf_counter()

        print('logging new conversation...')
        messId = 'user-'+str(uuid.uuid4())
        convId = 'conv-'+str(uuid.uuid4())

        print('convId: ', convId)

        def log(tx):
            tx.run("""
            create (c:Conversation)-[:FIRST]->(m:Message)
            set c.id = $convId, c.llm = $llm,
                c.temperature = $temperature,
                c.public = true,
                m.id = $messId, m.content = $content,
                m.role = $role, m.postTime = datetime(),
                m.embedding = $embedding,
                m.public = true
            
            with c
            merge (s:Session {id: $sessionId})
            on create set s.createTime = datetime()
            merge (s)-[:HAS_CONVERSATION]->(c)
                      """, convId=convId, llm=llm, messId=messId, 
                           temperature=st.session_state['temperature'],
                           content=user_input, role='user', sessionId=st.session_state['session_id'], 
                           embedding=st.session_state['recent_question_embedding'])
        
        # update the latest message in the log chain
        st.session_state['latest_message_id'] = messId

        try:
            with self.driver.session(database=self.database_name) as session:
                session.execute_write(log)
            
        except ConstraintError as err:
            print(err)

            session.close() 

        print('conversation init & user log time: '+str(round(time.perf_counter()-log_timer_start, 4))+" seconds.")

    def log_user(self, user_input):
        """
        This method logs a new user message to the neo4j database and 
        creates appropriate relationships.
        """

        log_timer_start = time.perf_counter()
        print('logging user message...')
        prevMessId = st.session_state['latest_message_id']
        messId = 'user-'+str(uuid.uuid4())

        def log(tx):
            tx.run("""
            match (pm:Message {id: $prevMessId})
            merge (m:Message {id: $messId})
            set m.content = $content,
                m.role = $role, m.postTime = datetime(),
                m.embedding = $embedding, m.public = true
                   
            merge (pm)-[:NEXT]->(m)
                      """, prevMessId=prevMessId, messId=messId, content=user_input, role='user',
                           embedding=st.session_state['recent_question_embedding'])
        
        # update the latest message in the log chain
        st.session_state['latest_message_id'] = messId

        try:
            with self.driver.session(database=self.database_name) as session:
                session.execute_write(log)
            
        except ConstraintError as err:
            print(err) 

        print('user log time: '+str(round(time.perf_counter()-log_timer_start, 4))+" seconds.")

    def log_assistant(self, assistant_output, context_indices):
        """
        This method logs a new assistant message to the neo4j database and 
        creates appropriate relationships.
        """

        log_timer_start = time.perf_counter()

        print('logging llm message...')
        prevMessId = st.session_state['latest_message_id']
        messId = 'llm-'+str(uuid.uuid4())

        def log(tx):
            tx.run("""
            match (pm:Message {id: $prevMessId})
            merge (m:Message {id: $messId})
            set m.content = $content,
                m.role = $role, m.postTime = datetime(),
                m.numDocs = $numDocs,
                m.vectorIndexSearch = true,
                m.prompt = $prompt,
                m.public = true
            merge (pm)-[:NEXT]->(m)

            with m
            unwind $contextIndices as contextIdx
            match (d:Document)
            where d.index = contextIdx

            with m, d
            merge (m)-[:HAS_CONTEXT]->(d)
                    """, prevMessId=str(prevMessId), messId=str(messId), content=str(assistant_output), 
                    role='assistant', contextIndices=context_indices, numDocs=st.session_state['num_documents_for_context'], prompt=st.session_state['general_prompt'])
            
        # update the latest message in the log chain
        st.session_state['latest_message_id'] = messId
        st.session_state['latest_llm_message_id'] = messId

        try:
            with self.driver.session(database=self.database_name) as session:
                session.execute_write(log)
            
        except ConstraintError as err:
            print(err) 

        print('assistant log time: '+str(round(time.perf_counter()-log_timer_start, 4))+" seconds.")

    def rate_message(self, rating: str):
        """
            This message rates an LLM message given a rating and uploads
            the rating to the database.
        """

        if 'latest_llm_message_id' in st.session_state:
            rate_timer_start = time.perf_counter()

            print('rating llm message...')
            print('updating id; ', st.session_state['latest_llm_message_id'])
            
            def rate(tx):
                tx.run("""
                match (m:Message {id: $messId})
    
                set m.rating = $rating
                        """, rating=rating, messId=st.session_state['latest_llm_message_id'])
                    
            try:
                with self.driver.session(database=self.database_name) as session:
                    session.execute_write(rate)
                
            except ConstraintError as err:
                print(err) 

            print('assistant rate time: '+str(round(time.perf_counter()-rate_timer_start, 4))+" seconds.")

    def create_prompt(self, question: str):
        """
        This function creates a prompt to be sent to the LLM.
        """

        context_timer_start = time.perf_counter()
        context = self.create_context(question)

        # use context docs in the prompt
        if st.session_state['num_documents_for_context'] > 0:

            print("Context creation total time: "+str(round(time.perf_counter()-context_timer_start, 4))+" seconds.")

            st.session_state['general_prompt'] = """
                    Follow these steps exactly: question
                    1. Read this question as an experienced graph data scientist at Neo4j: 
                    2. Read and summarize the following context documents, ignoring any that do not relate to the user question: {context[['url', 'text']].to_dict('records')}
                    3. Use this context and your knowledge to answer the user question.
                    4. Return your answer with sources.
                    """
            
            return [f"""
                    Follow these steps exactly:
                    1. Read this question as an experienced graph data scientist at Neo4j: {question} 
                    2. Read and summarize the following context documents, ignoring any that do not relate to the user question: {context[['url', 'text']].to_dict('records')}
                    3. Use this context and your knowledge to answer the user question.
                    4. Return your answer with sources.
                    """, 
                    list(context['index'])]
        # no context
        else:

            st.session_state['general_prompt'] = """
                    Follow these steps exactly:
                    1. Read this question as an experienced graph data scientist at Neo4j: {question} 
                    2. Use your knowledge to answer the user question.
                    3. Return your answer with sources if possible.
                    """
            
            return [f"""
                    Follow these steps exactly:
                    1. Read this question as an experienced graph data scientist at Neo4j: {question} 
                    2. Use your knowledge to answer the user question.
                    3. Return your answer with sources if possible.
                    """, 
                    list()]

    def init_llm(self, llm_type: str, temperature: float):
        """
        This function initializes an LLM for conversation.
        Each time the LLM type is changed, the conversation is reinitialized
        and history is lost.
        """

        match llm_type:
            case "chat-bison 2k":
                return ChatVertexAI(model_name='chat-bison',
                        max_output_tokens=1024, # this is the max allowed
                        temperature=temperature, # default temp is 0.0
                        top_p=0.95, # default is 0.95
                        top_k = 40 # default is 40
                       )
            case "chat-bison 32k":
                return ChatVertexAI(model_name='chat-bison-32k',
                        max_output_tokens=8192, # this is the max allowed 
                        temperature=temperature, # default temp is 0.0
                        top_p=0.95, # default is 0.95
                        top_k = 40 # default is 40
                       )
            case "GPT-4 8k":
                # Tokens per Minute Rate Limit (thousands): 10
                # Rate limit (Tokens per minute): 10000
                # Rate limit (Requests per minute): 60
                return AzureChatOpenAI(openai_api_version=openai.api_version,
                       openai_api_key = openai.api_key,
                       openai_api_base = st.secrets['openai_endpoint'],
                       deployment_name = st.secrets['gpt4_8k_name'],
                       model_name = 'gpt-4',
                       temperature=temperature) # default is 0.7
            case "GPT-4 32k":
                # Tokens per Minute Rate Limit (thousands): 30
                # Rate limit (Tokens per minute): 30000
                # Rate limit (Requests per minute): 180
                return AzureChatOpenAI(openai_api_version=openai.api_version,
                       openai_api_key = openai.api_key,
                       openai_api_base = st.secrets['openai_endpoint'],
                       deployment_name = st.secrets['gpt4_32k_name'],
                       model_name = 'gpt-4-32k',
                       temperature=temperature) # default is 0.7
            case _:
                raise ValueError

    def create_conversation(self, llm_type:str):
        """
        This function intializes a conversation with the llm.
        The resulting conversation can be prompted successively and will
        remember previous interactions.
        """
        create_conversation_timer_start = time.perf_counter()
        print("llm type: ", llm_type)
        llm = self.init_llm(llm_type, st.session_state['temperature'])

        res = ConversationChain(
            llm=llm,
            memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)
            ) 
        print("Create conversation time: "+str(round(time.perf_counter()-create_conversation_timer_start, 4))+" seconds.")

        return res