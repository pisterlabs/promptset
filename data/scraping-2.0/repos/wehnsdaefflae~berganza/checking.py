import logging
from pathlib import Path

from fastapi import FastAPI
from openai import InvalidRequestError
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from various.openai_request import Berganza

logging_path = "berganza.log"
logging.basicConfig(filename=logging_path,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

print(f"logging to {Path(logging_path).resolve()}.")
logger = logging.getLogger("Berganza")

berganza = Berganza("resources/config.json")

app = FastAPI()


class AskPayload(BaseModel):
    message: str


@app.post("/check_post/")
async def check_post() -> dict[str, str]:
    return {
        "reply":    "all fine",
        "info":     "all fine",
    }


@app.post("/ask/")
async def ask_berganza(body_json: AskPayload) -> dict[str, str]:
    try:
        print("received request")

        question = body_json.message

        info = "all fine"

        if question is None:
            answer = "Hast Du Deine Zunge verschluckt?"
            info = "too short"

        else:
            try:
                # answer = berganza.ask(question.strip()[-1000:], ip.replace(".", "-"))
                answer = berganza.ask(question.strip()[-1000:], "user")

            except InvalidRequestError as e:
                logger.error(str(e))
                print(str(e))
                answer = "Deine Worte ergeben für mich keinen Sinn."
                info = f"InvalidRequestError: {str(e):s}"

        print(answer)
        return {
            "reply":    answer,
            "info":     info,
        }

    except Exception as e:
        error_message = str(e)
        logger.error(error_message)
        return {
            "reply":    "Mir ist es gar nicht gut. Ich muss wohl einen Arzt aufsuchen.",
            "info":     error_message,
        }


app.mount("/", StaticFiles(directory="website", html=True), name="website")