import openai
from openai.embeddings_utils import get_embedding
import json
from copy import deepcopy
import tiktoken
from tqdm.notebook import tqdm
import re
import itertools
from sentence_transformers import SentenceTransformer, util
import numpy as np
from classes.OpenaiConnector import OpenaiConnector

class ChatSimilarity(OpenaiConnector):
    """
    Returns most similar chat thread

    Attributes:
    - json_file_path (str): Path to the JSON file containing chat data.
    - api_key (str): API key for OpenAI authentication.
    - nli_model: Neural model for natural language inference and similarity.
    - chat_data (dict): Data structure containing chat threads loaded from JSON.
    - indexed_data (dict): Processed chat threads with question-response pairs.
    - similar_query_prompt (str): Predefined prompt loaded from a file for generating similar queries.
    - relevence_comparision_prompt (str): Predefined prompt loaded from a file for relevance comparison.
    - chat_thread_similar_queries (dict): Holds similar queries for each chat thread.
    - index_mapping (dict): Mapping of chat embeddings to their titles for FAISS indexing.
    - embedding_matrix (numpy.ndarray): Matrix of embeddings to be indexed using FAISS.
    """
    def __init__(self, json_file_path, api_key, instruction_model='gpt-3.5-turbo-instruct',
                 embed_model='text-embedding-ada-002',
                 chat_model='gpt-4',
                 nli_model='paraphrase-MiniLM-L6-v2'):
        """
        Constructor for ChatSimilarity class.

        Args:
        - json_file_path (str): Path to the JSON file containing chat data.
        - api_key (str): API key for OpenAI authentication.
        - instruction_model (str): Name of the open AI model to be used for instruction
        - embed_model: Name of the open AI model to be used for embedding
        - chat_model (str): Name of the open AI model to be used for chat
        - nli_model (str): Name of the sentence_transformers model for natural language inference and similarity
        """
        super().__init__(api_key, instruction_model= instruction_model,
                         chat_model=chat_model,embed_model=embed_model)
        self.nli_model = SentenceTransformer(nli_model)  #good balance of speed and accuracy
        self.json_file_path = json_file_path
        self.chat_data = self._load_json()        
        self.indexed_data = self._process_chats()
        self.similar_query_prompt = self._load_prompt_from_file('prompts/similar_queries.txt')
        self.relevence_comparision_prompt = self._load_prompt_from_file('prompts/chat_relevance.txt')
        self.assitant_message = self._load_prompt_from_file('prompts/assistant_message.txt')
        # initialise class variables which will be set later
        self.chat_thread_similar_queries = None #generate_similar_queries()
        self.index_mapping = None #prepare_embeddings_for_faiss()
        self.embedding_matrix = None #prepare_embeddings_for_faiss()
        
    def _load_json(self):
        """
        Loads the chat data from the specified JSON file.
        """
        with open(self.json_file_path, 'r') as file:
            chat_data = json.load(file)
        return chat_data
    
    def _load_prompt_from_file(self,filepath):
        """Load the content of a file and return as a string."""
        with open(filepath, 'r') as file:
            content = file.read()
        return content
    def _process_chats(self):
        """
        Creates a new data structure to hold question-response pairs for each chat thread.
        """
        indexed_data = {}    
        for thread in self.chat_data:
            mapping = thread.get('mapping', {})
            question, response = None, None
            
            # Iterate through the mappings to find user and assistant messages
            for key, value in mapping.items():
                # Guard against None value for 'message'
                if value.get('message') is not None:
                    message_content = value['message'].get('content', {}).get('parts', [])
                    # Check role of the author and assign question or response
                    if value['message'].get('author', {}).get('role') == 'user':
                        question = ' '.join(message_content)  # join in case parts have multiple segments
                    elif value['message'].get('author', {}).get('role') == 'assistant':
                        response = ' '.join(message_content)
    
                    # If both question and response are found, add to indexed_data
                    if question and response:
                        if thread['title'] not in indexed_data:
                            indexed_data[thread['title']] = []
                        indexed_data[thread['title']].append((question, response))
                        question, response = None, None  # reset for the next pair
        return indexed_data
        
    @staticmethod
    def split_on_number_period(s):
        """
        Static method to split a string based on numbered list pattern.

        Args:
        - s (str): Input string.

        Returns:
        - list[str]: List of split parts.
        """
        # Split on the pattern
        parts = re.split(r'\n\d+\.', s)
        
        # Remove leading numbers followed by a period and filter out empty strings
        cleaned_parts = [re.sub(r'^\d+\.', '', part).strip() for part in parts]
        return [part for part in cleaned_parts if part]

    
    def generate_similar_queries(self, token_limit=2048,max_response_tokens=150,
                                 num_responses=5):
        """
        Generate similar queries/questions based on the overall theme of chat threads using OpenAI.

        Args:
        - token_limit (int): Maximum allowed tokens for the API call. Default is 2048.
        - max_response_tokens (int): Maximum number of tokens in the model's response.
        - num_responses (int): Number of different responses or queries to generate.

        Modifies:
        - self.chat_thread_similar_queries: Fills with the generated similar queries.
        """
        def prepare_model_response(response):
            # extract similar queries from each response
            split_response = [ChatSimilarity.split_on_number_period(single_response) for
                           single_response in response]
            # now response is a 2D list. We need to flatten it
            flattend_response = list(itertools.chain.from_iterable(split_response))
            return flattend_response
        # get all user chat titles
        
        chat_thread_similar_queries = {}
        
        for chat_thread_title in tqdm(self.indexed_data.keys()):
            # init empty list to store prompts after chunking of the chat
            similar_queries = []
            # init empty list to store all user queries
            user_queries = []
            # Extract the conversations for the given chat_thread_title
            conversations = self.indexed_data[chat_thread_title]
            
            # Prepare the context by considering the token limit
            context = ""
            for question, response in reversed(conversations):  # Start from the latest conversation
                # Form the conversation context
                conversation_context = f"User: {question}\nAssistant: {response}\n"
                # Check if adding the next conversation breaches the max token limit
                new_token_count = len(list(self.instruction_tokenizer.encode(self.similar_query_prompt+ context + conversation_context)))
                if new_token_count <= token_limit:
                    context += conversation_context
                    continue
                else:
                    # get similar queries for the current chunk
                    prompt = self.similar_query_prompt.format(chat_context=context)
                    context = ''
                    response = self.get_instructor_model_response(prompt,max_response_tokens=max_response_tokens,
                               token_limit=token_limit,num_responses=num_responses)# get similar queries
                    response = prepare_model_response(response) #format the model response
                    similar_queries.extend(response)
            
            # get similar queries for the last chunk
            prompt = self.similar_query_prompt.format(chat_context=context)        
            response = self.get_instructor_model_response(prompt,max_response_tokens=max_response_tokens,
                           token_limit=token_limit,num_responses=num_responses)# get similar queries
            response = prepare_model_response(response) #format the model response
            similar_queries.extend(response)
            chat_thread_similar_queries[chat_thread_title] = {'queries':similar_queries}
            # generate and store embeddings for all queries in chat_thread_similar_queries
            embeddings = [self.get_text_embedding(f'''{query}''') for query in similar_queries]
            # embeddings = [self.get_text_embedding('''{query}''') for query in similar_queries]
            # self.bert_model.encode(sentence)
            chat_thread_similar_queries[chat_thread_title]['embeddings'] = embeddings
        self.chat_thread_similar_queries = chat_thread_similar_queries
        

    def prepare_embeddings_for_faiss(self):
        """
        Process embeddings in preparation for FAISS indexing.

        Raises:
        - ValueError: If chat queries and their embeddings have not been generated.

        Modifies:
        - self.index_mapping: Fills with mapping of embeddings to chat titles.
        - self.embedding_matrix: Fills with the combined matrix of embeddings.
        """       
        if not self.chat_thread_similar_queries:
            raise ValueError("Chat queries & embeddings have not been generated. Please run generate_similar_queries() first.")            
        index_mapping = {}
        for idx,chat_title in tqdm(enumerate(self.chat_thread_similar_queries.keys()),
                                   total=len(self.chat_thread_similar_queries.keys())):
            
            embeddings = np.array(self.chat_thread_similar_queries[chat_title]['embeddings'])
            num_query_embeddings = embeddings.shape[0]
            index_keys = index_mapping.keys()
            if len(index_mapping.keys())>0:
                last_index_key = max(index_keys) + 1
            else:
                last_index_key = 0
            apend_mapping = {idx:chat_title for idx in range(last_index_key,last_index_key+num_query_embeddings)}
            index_mapping.update(apend_mapping)
            if idx == 0:
                embedding_matrix = embeddings
            else:
                embedding_matrix = np.concatenate([embedding_matrix,embeddings])
        self.index_mapping = index_mapping
        self.embedding_matrix = embedding_matrix
        
    def get_similar_chats(self,user_query,faiss_index):
        """
        Get chat threads that are similar to a user's query using FAISS.

        Args:
        - user_query (str): The user's input query.
        - faiss_index: FAISS index instance containing chat embeddings.

        Returns:
        - list[str] or str or None: List of matching chat titles or a single matching title or None if no match found.
        """
        user_query_embedding = self.get_text_embedding(user_query)
        user_query_embedding = np.array(user_query_embedding)
        similarity, indices = faiss_index.find_similar_querys(user_query_embedding, num_neighbors=10)
        
        min_similarity_cutoff = 0.8 #arbitrarily chosen
        # Compute similarity with other users for the current actor
        # similarity, indices = self.find_similar_users(actor_vector, num_neighbors=top_n)
        
        # Flatten the indices and similarity arrays
        indices = indices.flatten()
        similarity = similarity.flatten()
        
        # Drop -1 indices
        indices = indices[np.logical_and(~np.isnan(indices), similarity > min_similarity_cutoff)]
        
        if len(indices)<0: # nothing matched
            return None
        chat_titles = list(set([faiss_index.index_mapping_dict[idx] for idx in indices]))
        return chat_titles    

    def extract_latest_chat_context(self,chat_thread_titles,token_limit=2048):
        """
        Extracts the latest context from given chat threads, considering a token limit.

        Args:
        - chat_thread_titles (list[str]): List of chat thread titles to extract context from.
        - token_limit (int): Maximum allowed tokens for the context extraction. Default is 2048.

        Returns:
        - list[str]: List of chat contexts.
        """
        latest_chat_context = {}
        chat_history_prompts = []
        
        for chat_thread_title in tqdm(chat_thread_titles):
            # Extract the latest conversation for the given chat_thread_title
            conversations = self.indexed_data[chat_thread_title]
            
            # Start the context by providing the chat title and the first prompt/response
            question, response = conversations[0]
            context = f"{chat_thread_title}: User{question} \nAssistant:{response}"
            for question, response in reversed(conversations):  # Start from the latest conversation
                # Form the conversation context
                conversation_context = f"User: {question}\nAssistant: {response}\n"
                # Check if adding the next conversation breaches the max token limit
                # new_token_count = len(list(self.chat_tokenizer.encode(self.relevance_prompt + context + conversation_context)))
                new_token_count = len(list(self.chat_tokenizer.encode(context + conversation_context)))
                if new_token_count <= token_limit:
                    context += conversation_context
                    continue
                else:
                    break
            chat_history_prompts.append(f"{chat_thread_title}: {context}")
    
        return chat_history_prompts
        
    def prepare_model_messages(self,chat_history_prompts,user_query):
        """
        Prepares a structured message format for querying the OpenAI model.

        Args:
        - chat_history_prompts (list[str]): Chat histories to consider.
        - user_query (str): User's input query.

        Returns:
        - list[dict]: Structured messages for the model.
        """
        messages = [
                # {"role": "system", "content": "You are a helpful assistant"}, 
                {"role": "user", "content": self.relevence_comparision_prompt}, 
                {"role": "assistant", "content": self.assitant_message.replace('another','a')}]
        for chat_history_prompt in chat_history_prompts:
            messages.append({'role':'user','content':chat_history_prompt})
            messages.append({'role':'assistant','content':self.assitant_message})
        messages.append({'role':'user','content':f'''new query:{user_query}'''})
        return messages

    def get_valid_chat_title(self, user_query: str, chat_titles: list[str]) -> str:
        """
        Find the most similar text from a list of chat titles by comparing it to a user query.
        
        Args:
        - user_query (str): The user's input query.
        - chat_titles (list[str]): A list of chat titles to compare with the user's query.
        
        Returns:
        - str: The most similar chat title.
        """
        
        # Initialize the model
        
        
        # Get the embeddings
        user_query_embedding = self.nli_model.encode(user_query, convert_to_tensor=True)
        chat_title_embeddings = self.nli_model.encode(chat_titles, convert_to_tensor=True)
        
        # Compute cosine similarities
        cosine_scores = [util.pytorch_cos_sim(user_query_embedding, chat_title_embedding) for chat_title_embedding in chat_title_embeddings]
        
        # Get the index of the highest similarity score
        most_similar_index = np.argmax(cosine_scores)
        
        return chat_titles[most_similar_index]

    def return_most_similar_chat(self,user_query,similar_chat_titles:list):
        '''
        Function to finalise the chat thread from a candidate of chat threads
    
        Args:
        - user_query(str): The user query which is to be compared
        - similar_chat_titles(list) : list of candidate chat thread titles
    
        Return:
        - None if there is no matching chat thread, else the title of the most 
            similar chat thread
        '''
        if len(similar_chat_titles)==0:
            print("No chat threads are similar")
            return None
        elif len(similar_chat_titles)==1:
            print(f"{similar_chat_titles[0]} is the single candidate")
            return similar_chat_titles[0]
        else:            
            print(f"multiple candidate : {similar_chat_titles}")
        most_similar_chat_prompts = self.extract_latest_chat_context(similar_chat_titles)
        messages = self.prepare_model_messages(most_similar_chat_prompts,user_query)
        model_response = self.get_chat_model_response(messages, max_response_tokens=1, num_responses=1, temperature=0)[0]
        most_similar_chat_title = self.get_valid_chat_title(model_response, similar_chat_titles)
        return most_similar_chat_title