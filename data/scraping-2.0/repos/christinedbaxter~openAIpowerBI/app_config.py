import os
from dotenv import load_dotenv
import openai
from msal import PublicClientApplication

load_dotenv()

# Set up OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up Power BI authentication
TENANT_ID = os.getenv('TENANT_ID')

# Application (client) ID of app registration
CLIENT_ID = os.getenv("POWER_BI_CLIENT_ID")
# Application's generated client secret: never check this into source control!
CLIENT_SECRET = os.getenv("CLIENT_SECRET_ID")

# You can configure your authority via environment variable
# Defaults to a multi-tenant app in world-wide cloud
AUTHORITY = "https://login.microsoftonline.com/" + TENANT_ID

REDIRECT_PATH = "/getAToken"  # Used for forming an absolute URL to your redirect URI.
# The absolute URL must match the redirect URI you set
# in the app's registration in the Azure portal.

# You can find the proper permission names from this document
# https://docs.microsoft.com/en-us/graph/permissions-reference
SCOPE = ["https://analysis.windows.net/powerbi/api/.default"]
# SCOPE = ["Dataset.ReadWrite.All"]

app = PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
