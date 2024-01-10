import os
import openai
from dotenv import load_dotenv 
from openai.error import RateLimitError
import backoff
load_dotenv()



# /Users/shtlpmac002/Documents/reviews_sentiments_gpt4/raw_data/laptop/raw
class CreateEmbeddings():
    def __init__(self) -> None:      
        
        # self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        pass
    
    @backoff.on_exception(backoff.expo,RateLimitError)
    def get_embeddings(self,text):
        openai.api_type = os.getenv("OPENAI_API_TYPE")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_version= os.getenv("OPENAI_API_VERSION")
        openai.api_base=os.getenv("OPENAI_API_BASE")
        model=os.getenv("MODEL")
        engine=os.getenv("ENGINE")
        self.model=openai.Embedding(engine=engine)
        embeddings=self.model.create(engine=engine,input=text,model=model)
        embeddings=embeddings['data'][0]['embedding']

        # embeddings=self.model.encode(text,convert_to_tensor=True).tolist()
        
        return embeddings
    

    
