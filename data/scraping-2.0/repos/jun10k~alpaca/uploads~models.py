from __future__ import unicode_literals
from django.db import models
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus

import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://your-azure-region.api.cognitive.microsoft.com/"
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_API_VERSION"] = "your-api-version"


# Create your models here.
class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_processed = models.BooleanField(default=False)

    def digest(self):
        self.is_processed = True
        self.save()
        loader = TextLoader(self.document.path)
        loader.encoding = "ISO8859-1"
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        docs = text_splitter.split_documents(docs)
        if docs is not None and docs != []:
            embeddings = OpenAIEmbeddings(chunk_size=10)
            vector_db = Milvus.from_documents(
                docs,
                embeddings,
                connection_args={"host": "127.0.0.1", "port": "19530"},
            )

