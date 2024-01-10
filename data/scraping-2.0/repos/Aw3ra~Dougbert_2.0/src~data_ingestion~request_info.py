import pinecone
import os
import openai





# initialise OpenAI


# Function for creating vectors to store in Pinecone
# Takes in a prompt and returns a vector
def create_vector(prompt):
    vector = openai.Embedding.create(
            input=prompt,
            engine= "text-embedding-ada-002",
        )   ["data"][0]["embedding"]
    return vector

# Function to search for similar vectors
# Takes in a prompt and returns a list of similar vectors
def search_vectors(prompt):
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pinecone.Index(index_name=index_name)
    # Create a vector from the prompt
    vector = create_vector(prompt)
    # Search for similar vectors
    results = index.query(queries=[vector], top_k=5, include_metadata=True, namespace='solana-projects')
    top_result = results['results'][0]['matches'][0]  # Get the first match
    print(top_result['score'])
    if top_result['score'] > 0.8:
        return top_result
    else:
        return None

# FUnction to return the system message if something is found, or a default message if nothing is found
def return_system_message(query):
    pinecone_api = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # initialise Pinecone
    pinecone.init(api_key=pinecone_api, environment=pinecone_env)

    query = query[0]['content']
    print(query)
    result = search_vectors(query)
    if result is not None:
        system_message = {"role": "system", "content":
                                                        f"""\n
                                                        Your task is to pretend to be a really chill dude on the internet who help users find information, you have no association to the information you find. \n\n
                                                        You have found this information from topic: \"{result['metadata']['title']}\" \n\n
                                                        With a url of: \"{result['metadata']['link']}\" \n\n
                                                        Reply to the user using only this information from the project above, you are completely independent of the project: \"{result['metadata']['content']}\" \n\n
                                                        """}
    else:
        system_message = None
    
    return system_message



# Function that takes a result from search_vectors and creates a response to the request from the metadata using openAI
def create_response(query, result):
    if result is not None:
        system_message = {"role": "system", "content": 
                                                        f"""\n
                                                        Your task is to pretend to be a really chill dude on the internet who help users find information, you have no association to the information you find. \n\n
                                                        You have found this information from topic: \"{result['metadata']['title']}\" \n\n
                                                        With a url of: \"{result['metadata']['link']}\" \n\n
                                                        Reply to the user using only this information from the project above, you are completely independent of the project: \"{result['metadata']['content']}\" \n\n
                                                        """}
                                
        examples = [system_message, {"role": "user", "content": query}]
    else:
        system_message = {"role": "system", "content": "Your task is to pretend you know nothing. You will apologise and ask for some information regarding the topic."}
        examples = [system_message, {"role": "user", "content": query}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=examples,
        temperature=0.3,
        max_tokens=3000,
        top_p=1
    )
    return response['choices'][0]['message']['content']

def return_response(query):
    result = search_vectors(query)
    response = create_response(query, result)
    return response
    
    
if __name__ == "__main__":
    # Searching for similar vectors
    query = "Can I use solane CLI locally?? Provide your source."
    response = return_response(query)

    print("--------------------")
    print('Query: ', query)
    print("--------------------")
    print('Response: ', response)
    print("--------------------")
    print("--------------------")

