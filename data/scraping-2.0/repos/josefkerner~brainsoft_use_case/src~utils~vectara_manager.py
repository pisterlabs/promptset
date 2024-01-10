import json
import ssl
orig_sslsocket_init = ssl.SSLSocket.__init__
ssl.SSLSocket.__init__ = lambda *args, cert_reqs=ssl.CERT_NONE, **kwargs: orig_sslsocket_init(*args, cert_reqs=ssl.CERT_NONE, **kwargs)

from langchain.vectorstores import Vectara
from src.utils.secret_manager import SecretManager
from authlib.integrations.requests_client import OAuth2Session
import logging, requests
from typing import List, Dict
import os

class VectaraManager:
    def __init__(self):
        self.secret_manager = SecretManager()

        self.jwt_token = self._get_jwt_token(
            auth_url=self.secret_manager.oauth_url,
            app_client_id=self.secret_manager.vectara_client_id,
            app_client_secret=self.secret_manager.vectara_api_key
        )

    def _get_jwt_token(self,auth_url: str, app_client_id: str, app_client_secret: str):
        """Connect to the server and get a JWT token."""
        token_endpoint = f"{auth_url}/oauth2/token"
        session = OAuth2Session(
            app_client_id, app_client_secret, scope="")
        token = session.fetch_token(token_endpoint, grant_type="client_credentials")
        return token["access_token"]

    def get_vectara_doc_json(self, title: str, text : str, metadata: Dict):
        '''
        Returns vectara doc json
        :param title:
        :param text:
        :param metadata:
        :return:
        '''

        sections = [{
            'text': text,
            'source': metadata['document_id']

        }]
        document = {
            'title': title,
            'source': metadata['document_id'],
            'document_id': metadata['document_id'],
            'section': sections,
            'metadata_json': json.dumps(metadata),
            'metadata': json.dumps(metadata),
        }
        request = {
            'document': document,
            'corpus_id': self.secret_manager.corpus_id,
            'customer_id': self.secret_manager.customer_id
        }
        return request

    def upload_text_with_metadata(self, title: str, text: str, metadata: Dict):
        '''
        Uploads text document to vectara
        :param title:
        :param text:
        :param metadata:
        :return:
        '''


        post_headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "customer-id": self.secret_manager.customer_id
        }
        gen_doc = self.get_vectara_doc_json(
            title=title,
            text=text,
            metadata=metadata
        )

        idx_address = "https://api.vectara.io/v1/index"

        response = requests.post(
            idx_address,
            json=gen_doc,
            verify=True,
            headers=post_headers)

        if response.status_code != 200:
            logging.error("REST upload failed with code %d, reason %s, text %s",
                          response.status_code,
                          response.reason,
                          response.text)
            print(response.content)
            raise ValueError("REST upload failed")

        else:
            print("REST upload successful")

    def _get_upload_files(self, file: Dict):
        '''
        Returns vectara file payload json
        :param file:
        :return:
        '''

        return [("file", (file['base_filename'],
                          file['payload'],
                          "application/octet-stream"))]

    def upload_files(self, files: List[Dict]):
        '''
        Uploads file documents to vectara
        :param files:
        :return:
        '''
        self.jwt_token = self._get_jwt_token(
            auth_url=self.secret_manager.oauth_url,
            app_client_id=self.secret_manager.vectara_client_id,
            app_client_secret=self.secret_manager.vectara_api_key
        )

        post_headers = {
            'Authorization': f'Bearer {self.jwt_token}',
            'Content-Type': 'multipart/form-data',
            'Accept': 'application/json'
        }
        url = f"https://api.vectara.io/v1/upload?c={self.secret_manager.customer_id}&o={self.secret_manager.corpus_id}"
        for file in files:
            response = requests.post(
                url=url,
                data={},
                files=self._get_upload_files(file),
                verify=False,
                headers=post_headers)

            if response.status_code != 200:
                logging.error("REST upload failed with code %d, reason %s, text %s",
                              response.status_code,
                              response.reason,
                              response.text)
                print(response.content)
                raise ConnectionError(f"REST upload failed for file {file['base_filename']}")

            else:
                logging.info("REST upload successful")
                return True


    def get_vectara_client(self):
        self.jwt_token = self._get_jwt_token(
            auth_url=self.secret_manager.oauth_url,
            app_client_id=self.secret_manager.vectara_client_id,
            app_client_secret=self.secret_manager.vectara_api_key
        )
        os.environ['VECTARA_CUSTOMER_ID'] = self.secret_manager.customer_id
        os.environ['VECTARA_CORPUS_ID'] = str(self.secret_manager.corpus_id)
        os.environ['VECTARA_API_KEY'] = self.secret_manager.vectara2_api_key

        vectara = Vectara(
            vectara_customer_id=self.secret_manager.customer_id,
            vectara_api_key=self.secret_manager.vectara2_api_key,
            vectara_corpus_id=self.secret_manager.corpus_id
        )
        return vectara