import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import SupabaseVectorStore
from supabase import create_client

from src.domain.model.documents.documents_repository import DocumentsRepository


class SupabaseDocumentsRepository(DocumentsRepository):
    def __init__(self):
        self.supabase = create_client(
            os.environ.get("SUPABASE_API_URL"), os.environ.get("SUPABASE_API_KEY")
        )
        auth = self.supabase.auth.sign_in_with_password(
            {
                "email": os.environ.get("SUPABASE_GLOSSARY_EMAIL"),
                "password": os.environ.get("SUPABASE_GLOSSARY_PASSWORD"),
            }
        )
        self.supabase.postgrest.auth(auth.session.access_token)

        self.embeddings = OpenAIEmbeddings(
            model=os.environ.get("OPENAI_EMBEDDINGS_MODEL"),
            deployment=os.environ.get("OPENAI_EMBEDDINGS_DEPLOYMENT"),
            chunk_size=1,
        )

    def __del__(self):
        self.supabase.auth.sign_out()

    def from_with_query(
        self, docs: list[Document], query_name: str
    ) -> SupabaseVectorStore:
        return SupabaseVectorStore.from_documents(
            documents=docs,
            embedding=self.embeddings,
            client=self.supabase,
            table_name="documents",
            query_name=query_name,
        )

    def delete_all(self) -> None:
        self.supabase.table("documents").delete().neq("content", None).execute()
