# 챗봇을 통한 사용자의 재진단을 위하여 약속된 벡터 DB의 인덱스에 메타데이터와 함께 삽입
import os
import openai
from config import config
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = config.PINECONE_API_KEY
print(pinecone_api_key)

# 벡터 DB관련
from llama_index import SimpleDirectoryReader
import pinecone
from llama_index import GPTVectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore

def Insert_Violation():
    try:

        pinecone.init(api_key=pinecone_api_key, environment="asia-northeast1-gcp")
        pinecone_index = pinecone.Index("pdf-index")

        # 문서 로드
        documents = SimpleDirectoryReader("./pripen/VectorDB/data").load_data()

        # Pinecone 벡터 스토어 생성, 네임 스페이스는 pripen
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace='pripen')

        # 스토리지 컨텍스트 생성
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 문서에서 GPT 벡터 스토어 인덱스 생성
        GPTVectorStoreIndex.from_documents(documents=documents, storage_context=storage_context)

        # 벡터 스토어에서 GPT 벡터 스토어 인덱스 생성
        GPTVectorStoreIndex.from_vector_store(vector_store)

        return  True
    except ValueError as e:
        print(f"설정 오류: {e}")
        return False
    except openai.error.OpenAIError as e:
        print(f"OpenAI API 오류: {e}")
        return False
    except pinecone.core.PineconeError as e:
        print(f"Pinecone 관련 오류: {e}")
        return False
    except Exception as e:
        print(f"알 수 없는 오류: {e}")
        return False