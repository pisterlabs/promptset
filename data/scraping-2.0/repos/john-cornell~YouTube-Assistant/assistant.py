from dotenv import load_dotenv
from embeddings_store import get_embeddings, Embeddings

from langchain.chat_models.anthropic import ChatAnthropic
from chains.VideoTranscriptQueryChain import VideoTranscriptQueryChain
from chains.VideoTranscriptQueryConversationChain import VideoTranscriptQueryConversationChain

load_dotenv()

embeddings = get_embeddings(Embeddings.HUGGINGFACE)

class assistant:
    def __init__(self, k=20):
        chat = ChatAnthropic(temperature=0.4)
        self.chain = VideoTranscriptQueryConversationChain(llm=chat, k=k, embeddings=Embeddings.HUGGINGFACE, debug=True)

    def get_response_from_query(self, url : str, query : str):                        
        response = self.chain.run({"url": url, "query": query})

        output = response["response"]
        metadata = response["metadata"]

        return output, metadata["docs"], metadata["history"], metadata["prompt"]
