# .\venv\Scripts\activate
# pip install --upgrade --force-reinstall -r requirements.txt
# pipreqs ./ --force --encoding=utf8
"""
clickhouse version issue https://github.com/imartinez/privateGPT/issues/723
clickhouse-connect==0.5.22
python-multipart==0.0.6
"""
import chromadb
import dotenv
import openai
import pipreqs
import tiktoken
import uvicorn
