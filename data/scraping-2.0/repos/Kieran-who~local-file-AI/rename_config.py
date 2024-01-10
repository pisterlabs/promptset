"""

RENAME THIS FILE TO: config.py
(so remove 'rename_')

"""

# AZURE BACKUP FOR WEAVIATE -> OPTIONAL
AZURE_STORAGE_CONNECTION_STRING = ""
AZURE_CONTAINER_NAME = ""

# AZURE OPENAI -> OPTIONAL
AZURE_OPENAI_KEY = ""
AZURE_OPENAI_BASE_URL = ""
# Embed model for Azure OpenAI -> this is the name of the deployment for the text embedding
# If this is not set, the embedding will be retrieved from OpenAI
EMBED_MODEL = ''

# OPENAI KEY -> OPTIONAL ALTHOUGH EITHER THIS OR AZURE OPENAI NEEDS TO BE PROVIDED
OPEN_AI_KEY = "sk-"

# Whether this is the index machine or not -> if false, it will not index files but instead update the vector database from the backup (Only works if backup exists, this can be from any machine) Backups retrieved from the AZURE BACKUP FOR WEAVIATE configuration
INDEX_MACHINE = True

# INDEX SETTINGS
# The path of the folder to index
INDEX_PATH = ""
# Whether to check segments or not before adding to db. This calls chat-3.5-turbo for each segment and asks whether it should be saved to the db. Weeds out any gibberish or non-sensical segments that result from document headers or poor pdf extraction - however additional costs in the extra time to run and the API costs.
CHECK_CHUNKS = False
# any folder names within your directory to index that you want to ignore
FOLDERS_TO_IGNORE = [""]

# Default AI Models for summarisation; this must be set
# This needs to be updated based on your choice of Azure vs OpenAI. If Azure, add the deployment name. If OpenAI, use the standard model names they provide (your API key must be able to access them)
DEFAULT_SUMMARISATION_MODEL = ""  # e.g. if not using Azure: "gpt-3.5-turbo"
# if you are using Azure, there is a fallback option to make calls to OpenAI in the event the Azure API fails (e.g. rate limits or content filtering). Turn the fallback off by not adding an OpenAI key.
FALLBACK_OPEN_AI_MODEL = ""
