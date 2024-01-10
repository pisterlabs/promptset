# Create aoai class to connect to Azure OpenAI API and have a run function to send a prompt and get a response

class demo_utils:
    import os
    import openai
    import dotenv
    from openai.embeddings_utils import cosine_similarity
    import chromadb
    from chromadb.utils import embedding_functions


    dotenv.load_dotenv()
    openai.api_type = "azure"
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def __init__(self, engine: str="gpt4"):
        self.engine = engine
        self.client = None

    def run(self, prompt, temperature: float=0.5, max_tokens: int=500, 
                 top_p: float=0.5, 
                 frequency_penalty: float=0.0, 
                 presence_penalty: float=0.0,
                 stop: list=[],
                 isRaw: bool=False):
        response = self.openai.ChatCompletion.create(
            engine=self.engine,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )

        if isRaw:
            return response
        else:
            return response.choices[0].message['content']
        
    def similarity(self, text1, text2, engine: str="text-embedding-ada-002"):
        import numpy as np
        res = self.openai.Embedding.create(
            engine="text-embedding-ada-002",
            input=[text1, text2],
        )
        a = res["data"][0]["embedding"]
        b = res["data"][1]["embedding"]
        # return self.cosine_similarity(res["data"][0]["embedding"], res["data"][1]["embedding"])
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  

    
    def getorcreatecollection(self, db_name: str="compliance_kb", engine: str="text-embedding-ada-002"):
        # Using OpenAI Embeddings. This assumes you have the openai package installed
        openai_ef = self.embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.os.getenv("OPENAI_API_KEY"),
            model_name=engine
        )

        # setup Chroma in-memory, for easy prototyping. Can add persistence easily!
        self.client = self.chromadb.Client()

        # Create collection. get_collection, get_or_create_collection, delete_collection also available!
        return self.client.get_or_create_collection(db_name,  embedding_function=openai_ef)
    
    def deletecollection(self, db_name: str="compliance_kb"):
        self.client.delete_collection(db_name)
        print("Deleted collection: " + db_name)
        return self.getorcreatecollection()