from openai import OpenAI
from dotenv import load_dotenv # pip install python-dotenv
from pyzotero import zotero
import cmd
import os

load_dotenv()

zotero_api_key = os.environ.get("ZOTERO_API_KEY")
zotero_user_id = os.environ.get("ZOTERO_USER_ID")
zot = zotero.Zotero(zotero_user_id, 'user', zotero_api_key)