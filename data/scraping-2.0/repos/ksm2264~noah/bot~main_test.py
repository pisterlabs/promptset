import os
import subprocess
import sys
import uuid
from bot.feature import implement_feature
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Get the input argument from the command line
input_str = sys.argv[1]

# Define the name of the new branch as a UUID
new_branch_name = str(uuid.uuid4())

implement_feature(input_str)
