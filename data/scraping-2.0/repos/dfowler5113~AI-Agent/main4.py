from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import BaseCallbackHandler
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from datetime import datetime
from supabase import create_client, Client
from InstructorEmbedding import INSTRUCTOR
from collections import deque
import warnings
import torch
import numpy as np
import faiss

warnings.filterwarnings("ignore")


Client = create_client("your_info", "your_info")


model = INSTRUCTOR('hkunlp/instructor-xl')

# Set up FAISS index
dimension = 768  # Assuming the embeddings are 768-dimensional. Adjust as needed.
index = faiss.IndexFlatL2(dimension)

# Fetch all data from the database
database_response = Client.table('history').select('embedding, user_input, llm_output').execute()
database_data = database_response.data


# Add all database embeddings to the FAISS index
database_embeddings = []
for entry in database_data:
    entry_embedding = np.array(entry['embedding'][0])[np.newaxis, :]
    index.add(entry_embedding)
    database_embeddings.append(entry)


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        pass

past_text = deque(maxlen=2)
callback_manager = CallbackManager(handlers=[MyCustomHandler])
longllm = LlamaCpp(
    model_path="your_info",
    callback_manager=callback_manager,
    max_tokens=50,
    temperature=1,
    top_p=0.8,
    n_ctx = 512, 
    logprobs=None,
    repeat_penalty=1.3,
    top_k=10,
    last_n_tokens_size=32,
    n_batch=500,
    verbose=False,
    use_mlock=True,
    n_threads=14,
)
longllm.client.verbose = False
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

while True:
    user_input = input("\nPlease enter something: ")
    user_input_vector = model.encode([user_input])[0]

    # Add the user input to the 'text' variable
    text = user_input

    # Print the content of 'text'
    history = " ".join(past_text)

    # First phase decision making
    decision_template = "Q: Does the following sentence request information from Wikipedia: {question}\n\n### A:"
    prompt = PromptTemplate(template=decision_template, input_variables=["question"])
    llm_chain_decision = LLMChain(prompt=prompt, llm=longllm)
    decision = llm_chain_decision.predict(question=text)

    print("Decision: " + decision)

    # Depending on decision, either fetch information from Wikipedia or generate a response
    if "yes" in decision.lower():
        # Fetch from Wikipedia
        wiki_test = wikipedia.run(text)

        # Split Wikipedia text into chunks of 300 tokens
        wiki_test_tokens = wiki_test.split()
        wiki_test_chunks = [
            wiki_test_tokens[i: i + 300] for i in range(0, len(wiki_test_tokens), 300)
        ]

        chunk_responses = []  # stores all chunk responses
        for wiki_test_chunk in wiki_test_chunks:
            wiki_test = ' '.join(wiki_test_chunk)
            
            # Now use the LLM to generate a response using both the user's input and the Wikipedia data
            response_template = "Given the information I found on Wikipedia: {wiki_info}, and your query: {user_query}, here's my response:\n\n### A:"
            prompt = PromptTemplate(template=response_template, input_variables=["wiki_info", "user_query"])
            llm_chain_response = LLMChain(prompt=prompt, llm=longllm)
            chunk_response = llm_chain_response.predict(wiki_info=wiki_test, user_query=text)
           
            chunk_responses.append(chunk_response)

        # Join the chunk responses into a single string
        all_responses = " ".join(chunk_responses)

        # Make sure the total tokens do not exceed the model's limit
        if len(all_responses.split()) > 300:
            all_responses = " ".join(all_responses.split()[:300])

        # Use the model to generate a final, cohesive response
        cohesive_template = "Based on my previous thoughts: {last_responses}, here's a final cohesive response:\n\n### A:"
        prompt = PromptTemplate(template=cohesive_template, input_variables=["last_responses"])
        llm_chain_cohesive = LLMChain(prompt=prompt, llm=longllm)
        cohesive_response = llm_chain_cohesive.predict(last_responses=all_responses)
        past_text.append(f"LLM Response: {cohesive_response}.")
       
        print("\nResponse: " + cohesive_response)

        # Get embeddings 
        embeddings = model.encode([[cohesive_response, text]])
        embeddings_np = np.array(embeddings.tolist())

        # Insert into Supabase
        Client.table('history').insert({
        'embedding': embeddings.tolist(),
        'user_input': user_input,
        'llm_output': cohesive_response,   # Use user_input as a placeholder
        'timestamp': datetime.now().isoformat()   # current time
    }).execute()

        # Update FAISS index
        embeddings_np = np.array(embeddings.tolist())
        if len(embeddings_np.shape) == 1:
            embeddings_np = embeddings_np[np.newaxis, :]  # Convert to 2D array if necessary

        index.add(embeddings_np)
        # Use FAISS to find similar past interactions
        D, I = index.search(user_input_vector[np.newaxis, :], 5)

        # Retrieve the corresponding past interactions
        similar_interactions = [database_embeddings[i] for i in I[0] if i < len(database_embeddings)]


        # Add the similar interactions to the LLM's context
        for interaction in similar_interactions:
            past_text.append(f"Similar interaction - User: {interaction['user_input']}, LLM: {interaction['llm_output']}")
            

    else:
        
        long_template = "Order: In light of our conversation {history}, your thoughts or continuance of our dialogue would be appreciated, try to keep the response to under twenty words. Here's my statement: {question}\n\n### Followed order Response:"
        prompt = PromptTemplate(template=long_template, input_variables=["history", "question"])
        llm_chain_long = LLMChain(prompt=prompt, llm=longllm)
        response = llm_chain_long.predict(history=history, question=text)
        past_text.append(f"Your Response: {text}.")
        past_text.append(f"My Response: {response}.")
        
        print("\nResponse: " + response)

       
        embeddings = model.encode([[response, text]])
        embeddings_np = np.array(embeddings.tolist())

      
        Client.table('history').insert({
        'embedding': embeddings.tolist(),
        'user_input': user_input,
        'llm_output': response,   
        'timestamp': datetime.now().isoformat()   #
    }).execute()
        
        embeddings_np = np.array(embeddings.tolist())
        if len(embeddings_np.shape) == 1:
            embeddings_np = embeddings_np[np.newaxis, :]  

        index.add(embeddings_np)

       
        D, I = index.search(user_input_vector[np.newaxis, :], 5)

       
        similar_interactions = [database_embeddings[i] for i in I[0] if i < len(database_embeddings)]


       
        for interaction in similar_interactions:
            past_text.append(f"Similar interaction - User: {interaction['user_input']}, LLM: {interaction['llm_output']}")
            
