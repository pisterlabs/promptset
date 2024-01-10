from typing import Any, Optional
from django.core.management.base import BaseCommand, CommandParser
from django.utils.encoding import force_str
from django.conf import settings
from adey_apps.users.models import User
from adey_apps.rag.models import Resource, Chat
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

class Command(BaseCommand):
    help = "Build a pg vector index for the user"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("email", type=str, help="Email of the user")
        parser.add_argument("chat_slug", type=str, help="Chat identifier slug")
    
    def handle(self, *args: Any, **options: Any) -> None:
        email = options.get("email", None)
        chat_slug = options.get("chat_slug", None)
        try:
            user = User.objects.get(email=email)
            self.stdout.write(f"User identifier: {user.identifier}")
            chat = Chat.objects.get(user=user, slug=chat_slug)
            resource = chat.resource_set.first()
            if resource:
                loader = TextLoader(resource.document.path)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = loader.load_and_split(text_splitter)
                self.stdout.write(f"User identifier: {os.environ.get('OPENAI_API_KEY', 'tt')}")
                embedding = OpenAIEmbeddings()
                db = PGVector.from_documents(
                    embedding=embedding,
                    documents=docs,
                    collection_name=force_str(chat.identifier),
                    connection_string="postgresql://adey_backend:secret@db:5432/adey_backend",
                )
                print(db)
        except User.DoesNotExist:
            self.stderr.write(f"No such user exists with the given email address. Error: {e.__str__()}")
        except Chat.DoesNotExist:
            self.stderr.write(f"No such Chat exists with the given slug. Error: {e.__str__()}")
        except Exception as e:
            self.stderr.write(f"Exception occurred. Error: {e.__str__()}")

        