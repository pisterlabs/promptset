import os
import shutil
import json
import asyncio
import qdrant_client
from langchain import PromptTemplate
from integrations import discord_integration, slack_integration


class Config:
    def __init__(self):
        self.db_client = None
        config_path = "/etc/enhancedocs"
        data_path = "/data/enhancedocs"
        config_file_name = "config.json"
        config_file_path = config_path + "/" + config_file_name
        os.makedirs(config_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)
        if not os.path.exists(config_file_path):
            shutil.copyfile(config_file_name, config_file_path)
        with open("prompt_template.txt", 'r') as f:
            prompt_template = f.read()
            self.prompt = PromptTemplate(
                template=prompt_template, input_variables=["summaries", "question", "project_name"]
            )
        with open(config_file_path, 'r') as f:
            config_json = json.load(f)
            self.docs_base_url = config_json.get("docs_base_url")
            self.project_name = config_json.get("project_name")
            integrations = config_json.get("integrations")
            if integrations:
                if integrations.get("discord"):
                    asyncio.create_task(discord_integration.start())
                if integrations.get("slack"):
                    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
                    slack_signing_secret = os.environ.get("SLACK_SIGNING_SECRET")
                    if slack_bot_token is not None and slack_signing_secret is not None:
                        self.slack_client = slack_integration.SlackClient(slack_bot_token, slack_signing_secret)

        self.vector_index_file_path = data_path + "/vectorstore.index"
        self.vector_store_file_path = data_path + "/vectorstore.pkl"
        self.default_collection_name = "enhancedocs"
        self.qdrant_url = os.environ.get("QDRANT_URL")
        self.qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        self.qdrant_args = {}
        if self.qdrant_url:
            self.qdrant_args["url"] = self.qdrant_url
            if self.qdrant_api_key:
                self.qdrant_args["api_key"] = self.qdrant_api_key
            self.db_client = qdrant_client.QdrantClient(**self.qdrant_args)
        else:
            self.qdrant_args["path"] = data_path + "/qdrant.local"

    # TODO: This should be replaced with qdrant local, but seems have an issue. will switch once fixed
    def is_external_db_used(self):
        return self.db_client is not None
