import json
import os
from iterative import get_config as _get_config
from iterative.models.assistant import IterativeAssistant
from openai import OpenAI
from logging import getLogger

logger = getLogger(__name__)


class AssistantManager:
    def __init__(self, client):
        self.client: OpenAI = client

    def list_assistants(self):
        try:
            assistants = self.client.beta.assistants.list()
            return assistants.data
        except Exception as e:
            print(f"Error listing assistants: {e}")
            return None

    def get_assistant(self, asst_id=None):
        try:
            if not asst_id:
                asst_id = _get_config().get("assistant_id")

            assistant = self.client.beta.assistants.retrieve(asst_id)
            return assistant
        except Exception as e:
            print(f"Error getting assistant: {e}")
            return None

    def get_assistant_info(self, asst_id=None):
        try:
            if not asst_id:
                asst_id = _get_config().get("assistant_id")
            assistant = self.client.beta.assistants.retrieve(asst_id)
            return assistant.json()
        except Exception as e:
            print(f"Error getting assistant: {e}")
            return None
        
    def update_assistant(self, asst_id=None, **kwargs):
        print("Updating assistant")

        if not asst_id:
            asst_id = _get_config().get("assistant_id")
        
        if not asst_id:
            print("No assistant ID provided.")
            return 
        
        # Get the schema of IterativeAssistant to know the valid keys
        valid_keys = json.loads(IterativeAssistant.schema_json())['properties'].keys()
        
        # Filter kwargs to only include valid keys
        attrs = {key: kwargs[key] for key in valid_keys if key in kwargs}
        if 'id' in attrs:
            del attrs['id']

        if 'created_at' in attrs:
            del attrs['created_at']
        
        if 'object' in attrs:
            del attrs['object']

        # Special handling for 'tools', if present in kwargs
        if 'tools' in kwargs:
            tools = kwargs['tools']
            action_cap = _get_config().get("actions_cap", 128)
            if len(tools) > action_cap:
                attrs['tools'] = tools[:action_cap]
                logger.warning(f"More than {action_cap} tools provided, truncating to {action_cap}.")
            else:
                attrs['tools'] = tools

        # TODO: Put this back in after fixing tools to always include retrieval 
        if 'file_ids' in kwargs:
            del attrs['file_ids']

        # Update the assistant if there are valid attributes to update
        if attrs:
            try:
                logger.info(f"Updating assistant {asst_id} with attributes: {attrs.keys()}")
                # remove all keys and values where the value is none
                attrs = {k: v for k, v in attrs.items() if v is not None}
                assistant = self.client.beta.assistants.update(asst_id, **attrs)
                logger.info(f"Assistant {asst_id} updated.")
                return assistant
            except Exception as e:
                logger.error(f"Error updating assistant: {e}")
                return None
        else:
            logger.info(f"No valid attributes to update for assistant {asst_id}.")
            return None
        
    def upload_docs_folder(self, folder_path="docs"):
        """Uploads the contents of the specified 'docs' folder to OpenAI."""
        uploaded_files = []
        try:
            if not os.path.exists(folder_path):
                print(f"Folder '{folder_path}' does not exist.")
                return None

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    uploaded_file = self.upload_document(file_path)
                    if uploaded_file:
                        uploaded_files.append(uploaded_file)
            
            print("All documents uploaded successfully.")
            return uploaded_files
        except Exception as e:
            print(f"Error uploading documents: {e}")
            return None

    def upload_document(self, file_path):
        """Uploads a single document to OpenAI."""
        try:
            with open(file_path, 'rb') as file:
                response = self.client.files.create(file=file, purpose="assistants")
                print(f"Document '{file_path}' uploaded successfully.")
                return response
        except Exception as e:
            print(f"Error uploading document '{file_path}': {e}")
            return None

    def delete_file(self, file_id):
        """Deletes a file from OpenAI."""
        try:
            self.client.files.delete(file_id)
            print(f"File with ID '{file_id}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting file '{file_id}': {e}")

    def retrieve_file(self, file_id):
        """Retrieve a specific file from OpenAI."""
        try:
            self.client.files.list()
            return self.client.files.retrieve(file_id)
        except Exception as e:
            print(f"Error retrieving file '{file_id}': {e}")
            return None
        
    def retrieve_files(self):
        """Retrieve a specific file from OpenAI."""
        return self.client.files.list()

    def retrieve_file_content(self, file_id):
        """Retrieve the content of a specific file from OpenAI."""
        try:
            return self.client.files.retrieve_content(file_id)
        except Exception as e:
            print(f"Error retrieving content for file '{file_id}': {e}")
            return None

    def list_files(self):
        """List all files uploaded to OpenAI."""
        try:
            return self.client.files.list()
        except Exception as e:
            print("Error listing files:", e)
            return None

    def set_files_to_assistant(self, assistant_id, file_ids):
        """Sets specified files to the assistant, removing any previous files."""
        try:
            # Remove previous files
            existing_files = self.get_assistant_info(assistant_id).get('file_ids', [])
            for file_id in existing_files:
                self.delete_file(file_id)

            # Update assistant with new files
            self.update_assistant(assistant_id, {'file_ids': file_ids})
            print(f"Files set to assistant {assistant_id}")
        except Exception as e:
            print(f"Error setting files to assistant '{assistant_id}': {e}")
