
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API key, base, type, and version for openai

def setOpenAIUsEnvironmentVariable():
    openai.api_base = os.getenv("API_BASE_US")
    openai.api_type = os.getenv("AZURE_API_TYPE")
    openai.api_version = os.getenv("API_VERSION_US")
    openai.api_key = os.getenv("API_KEY_US") 
    
def setOpenAIEuEnvironmentVariable():
    openai.api_key = os.getenv("API_KEY_EU_CLOUDMELTER")
    openai.api_base = os.getenv("API_BASE_EU_CLOUDMELTER")
    openai.api_type = os.getenv("AZURE_API_TYPE")
    openai.api_version = os.getenv("AZURE_API_VERSION_EU_CLOUDMELTER")
    
def setOpenAIEuEnvironmentVariableSmall():
    openai.api_key = os.getenv("API_KEY_EU_CLOUDMELTER")
    openai.api_base = os.getenv("API_BASE_EU_CLOUDMELTER")
    openai.api_type = os.getenv("AZURE_API_TYPE")
    openai.api_version = os.getenv("AZURE_API_VERSION_EU_CLOUDMELTER")