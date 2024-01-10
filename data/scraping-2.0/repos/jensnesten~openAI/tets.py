import os
import openai
openai.organization = "org-9RZKTW2xMPGSo2H6BWgzrDTp"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()