import openai
import utils
from fastapi import FastAPI

app = FastAPI()

@app.post("/memorize")
async def add_message(message: str, speaker: str, timestamp: float):
    print('\n\nMEMORIZING -', message)
    utils.memorize(message, speaker, timestamp)
    return {"detail": "Memorized!"}

@app.get("/remember")
async def search():
    print('\n\nRETRIEVING TEMPORARY MEMORY')
    return {"results": utils.handle_corpus()}
