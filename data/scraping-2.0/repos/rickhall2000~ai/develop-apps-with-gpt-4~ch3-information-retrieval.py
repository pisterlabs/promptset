# Creating an expert from a long pdf document

# Two options are (covered in a later chapter):
    # Fine-tuning retraining an existing model on a specific dataset
    # and Few-shot learning Adding examples to the prompt to send to the model

# The Approach in this example is a combination of 3 services
  # 1. The user's query is handled by an intent service, that uses OpenAI to understand the user's intent
  # 2. The intent is sent to an information retreival service that compares embeddings of the intent
    #    with embeddings of the document's sentences, stored in a vector store
  # 3. A response service that takes the output from the datbase and uses open ai to generate a response
  
# The example is here: https://github.com/malywut/gpt_examples 

# What follows is not a working model, just notes from the chapter for understanding

import openai


class DataService():
    def __init__(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            pasword=REDIS_PASSWORD
        )
        
    def pdf_to_embeddings(self, pdf_path: str, chunk_length: int = 1000):
        # this chunks based on characters, paragraphs would be better
        reader = PdfReader(pdf_path)
        chunks = []
        for page in reader.pages:
            text_page = page.extract_text()
            chunks.extend([text_page[i:i+chunk_length]
                        for i in range(0, len(text_page), chunk_length)])
        # Create embeddings
        response = openai.Embedding.create(model="text_embedding-ada-002",
                                        input=chunks)
        
        return [{'id': value['index'],
                'vector' :value['embedding'],
                'text' :chunks[value['index']]} for value in response['values']]
        
    def load_data_to_redis(self, embeddings):
        for embedding in embeddings:
            key = f"{PREFIX}:{str(embedding['id'])}"
            embeddings["vector"] = np.array( 
                                embedding["vector"], dtype=np.float32).tobytes()
            self.redis_client.hset(key, mapping=embedding) 

    def search_redis(self, user_query: str):
        embedded_query=openai.Embedding.create(
            input=user_query,
            model="text_embedding-ada-002")["data"][0]["embedding"]
        
        results = self.redis_client.ft(index_name).search(query, params_dict)
        return [doc['text'] for doc in results.docs] 

class IntentService():
    def __init__(self):
        pass 
    def get_indent(self, user_question: str):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": f"""Extract the keywords from the following
                 question: {user_question}"""}
            ]
        )
        return response["choices"][0]["message"]["content"]
    
class ResponseService():
    def __init__(self):
        pass 
    def generate_response(self, facts, user_question):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": f"""Based on the FACTS, answer the QUESTION.
                 QUESTION: {user_question}. FACTS: {facts}"""}
            ]
        )
        return (response['choices'][0]['message']['content'])
    
def run(question: str, file: str='ExplorersGuide.pdf'):
    data_service = DataService()
    data = data_service.pdf_to_embeddings(file)
    data_service.load_data_to_redis(data)

    intent_service = IntentService()
    intents = inten_service.get_intent(question)
    
    facts = service.search_redis(intents)
    
    return response_service.generate_response(facts, question)