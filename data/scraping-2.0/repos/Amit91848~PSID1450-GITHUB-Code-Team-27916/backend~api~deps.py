from typing import Generator, Annotated
from sqlmodel import Session
import os

from db.engine import engine

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from models import User, TokenPayload
from jose import jwt, JWTError
from core.settings import settings
from core.security import ALGORITHM
from pydantic import ValidationError
from core.services.file.UploadService import LocalUploadService, FileUploadService, S3Service
from core.services.vector_store.service import VectorStoreService, PineconeService, ChromaService
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from core.services.embedding.openai import get_openai_embeddings
from langchain_core.embeddings import Embeddings
from core.services.prakat.models import FlanT5_CT2, Generator_CT2
from core.services.prakat.llama import LocalModel


def get_db() -> Generator:
    with Session(engine) as session:
        yield session


reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"/api/login/access-token"
)


def create_upload_service():
    if settings.S3_ACCESS_KEY and settings.S3_SECRET_ACCESS_KEY:
        # s3_client = ...  # Instantiate your S3 client (boto3, aioboto, etc.)
        return S3Service()
    else:
        return LocalUploadService(local_storage_path=os.path.join(os.getcwd(),"files"))


def get_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def create_vector_store_service():
    if settings.PINECONE_API_KEY and settings.PINECONE_ENVIRONMENT:
        return PineconeService()
    else:
        print("returning chroma store")
        return ChromaService()

def create_grammar_check_model_service():
    return FlanT5_CT2(model_path=os.path.join(os.getcwd(),"models", "t5_grammarcheck_ct2"), tokenizer_path=os.path.join(os.getcwd(), "models", "flan_t5_base_tokenizer"), model_name="t5_grammarcheck_ct2")

def create_summarizer_model_service():
    return FlanT5_CT2(model_path=os.path.join(os.getcwd(),"models", "t5_summarizer_ct2"), tokenizer_path=os.path.join(os.getcwd(), "models", "flan_t5_base_tokenizer"), model_name="t5_summarizer_ct2")

def create_rag_model_service():
    return Generator_CT2(model_path=os.path.join(os.getcwd(), "models", "phi-1_5-ct2-int8"), tokenizer_path=os.path.join(os.getcwd(), "models", "phi-1_5-tokenizer"), model_name="phi-1_5-ct2-int8")

SessionDep = Annotated[Session, Depends(get_db)]
TokenDep = Annotated[str, Depends(reusable_oauth2)]
FileUploadServiceDep = Annotated[FileUploadService, Depends(
    create_upload_service)]
EmbeddingDep = Annotated[Embeddings, Depends(get_embeddings)]
VectorStoreDep = Annotated[VectorStoreService,
                           Depends(create_vector_store_service)]


def get_current_user(session: SessionDep, token: TokenDep) -> User:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=ALGORITHM)
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Could not validate credentials")

    user = session.get(User, token_data.sub)
    if not user:
        print("Cookie user not found")
        raise HTTPException(status_code=404, detail="User not found")
    return user


def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


CurrentUser = Annotated[User, Depends(get_current_active_user)]


def get_current_active_superuser(current_user: CurrentUser) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=400, detail="The user doesn't have enough privileges"
        )
    return current_user
