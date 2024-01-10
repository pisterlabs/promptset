import os
import confuse
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore import Client, CollectionReference
import openai

class LangameClient:
    def __init__(self, path_to_config_file: str = './config.yaml'):
        conf = confuse.Configuration('langame', __name__)
        conf.set_file(path_to_config_file)

        os.environ["HUGGINGFACE_KEY"] = conf["hugging_face"]["token"].get()
        os.environ["HUGGINGFACE_TOKEN"] = conf["hugging_face"]["token"].get()
        openai.api_key = conf["openai"]["token"].get()
        openai.organization = conf["openai"]["organization"].get()
        assert openai.api_key, "OPENAI_KEY not set"
        assert openai.organization, "OPENAI_ORG not set"
        assert os.environ.get("HUGGINGFACE_TOKEN"), "HUGGINGFACE_TOKEN not set"
        assert os.environ.get("HUGGINGFACE_KEY"), "HUGGINGFACE_KEY not set"

        # Firestore
        cred = credentials.Certificate(
            f'{os.path.dirname(path_to_config_file)}/{conf["google"]["service_account"]}')
        firebase_admin.initialize_app(cred)
        self._firestore_client: Client = firestore.client()
        self._memes_ref: BaseCollectionReference = self._firestore_client.collection(
            u"memes")

        #self._is_dev = "prod" not in (conf["google"]["service_account"])

    def purge(self, collection, sub_collections = []):
        def delete_collection(coll_ref, batch_size=20):
            docs = coll_ref.limit(batch_size).stream()
            deleted = 0

            for doc in docs:
                for sub in sub_collections:
                    for e in doc.reference.collection(sub).stream():
                        e.reference.delete()
                doc.reference.delete()
                deleted = deleted + 1

            if deleted >= batch_size:
                print(f'Deleted a batch of {deleted} {coll_ref.parent}')
                return delete_collection(coll_ref, batch_size)
        delete_collection(self._firestore_client.collection(collection))

