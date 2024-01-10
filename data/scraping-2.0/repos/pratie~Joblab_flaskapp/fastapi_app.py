from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import openai
import uuid
import os
import requests
import json
import tempfile
import config
import db

app = FastAPI()
templates = Jinja2Templates(directory="templates")
database_file = "database.json"
database = db.load(database_file)
settings = config.load("settings.json")

# ... [Converted FastAPI routes]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import openai
import uuid
import os
import requests
import json
import tempfile
import config
import db

app = FastAPI()
templates = Jinja2Templates(directory="templates")
database_file = "database.json"
database = db.load(database_file)
settings = config.load("settings.json")

# ... [Converted FastAPI routes]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
