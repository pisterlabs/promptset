import os
import json

from fastapi import APIRouter
from pydantic import BaseModel
from google.oauth2 import service_account
from google.cloud import vision
import openai

import util.myfirebase as myfirebase

# import util.vision_api_text_annote as vision_annote

router = APIRouter(prefix="/prescription-digitization", tags=["prescription"])


class Item(BaseModel):
    uid: str
    img_url: str

from dotenv import load_dotenv

load_dotenv(dotenv_path="configs/.env")

openai.api_key = os.getenv("OPENAI_KEY")

os.environ["GOOGLE_APPLICATION_CREDENTIALS_VISION"] = "configs/gCloud.json"
GOOGLE_APPLICATION_CREDENTIALS = os.environ["GOOGLE_APPLICATION_CREDENTIALS_VISION"]

try:
    VISION_API_CREDENTIALS = service_account.Credentials.from_service_account_file(
        GOOGLE_APPLICATION_CREDENTIALS
    )
except Exception as e:
    VISION_API_CREDENTIALS = service_account.Credentials.from_service_account_info(
        GOOGLE_APPLICATION_CREDENTIALS
    )
    print(e)


def detect_handwriting(img_url: str):
    client = vision.ImageAnnotatorClient(credentials=VISION_API_CREDENTIALS)

    image = vision.Image()
    image.source.image_uri = img_url
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    # response = vision_annote.annote_text(client=client, img_url=img_url)
    text = response.full_text_annotation.text

    if not text:
        raise Exception("Unable to locate text within this image.")
    return text


def generate_json(detected_text: str):
    prompt = """Generate a JSON representation of doctor information and prescriptions. If any data is missing or not available, use null as the value. The JSON should include the following fields:
    "prescribed_date": Date indicating when this prescription was issued by the doctor
    "hospital" : An object which contains information about hospital or clinic.
    "doctors": An array of objects representing doctors information such as name, qualification.
    "patient": An object which contains information about patient such as name, age, address, phone.
    "observations": An object for recording a doctor's findings and observations about patient, including information such as pulse, blood pressure, temperature, tongue condition, vomiting, pain level, symptoms, and other relevant details.
    "tests": An array of objects which contains information test advised to the patient by doctor.
    "medicationList": An array of objects representing medications. Each object should include the following fields:
        "medicineName": The name of the prescribed medicine that is an actual medicine.
        "dose": It contains information when to take this medication.
        "duration": This medication's duration in days. How long should patient take this medication.
        "route": A medication administration route is often classified by the location at which the drug is administered, such as orally or intravenous (injection).
    "otherInfo": An object which includes rest of the information of the prescription.
The given information is: {}
""".format(
        detected_text
    )

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    return json.loads(chat_completion.choices[0].message.content)


@router.post("/")
async def main(item: Item):
    try:
        uid = item.uid
        img_url = item.img_url

        text = str(detect_handwriting(img_url=img_url))
        # print(text)
        content = generate_json(text)

        doc_id = myfirebase.saveReport(
            uid,
            {
                "title": "Prescription Digitization",
                "img_url": img_url,
                "text": text,
                "output": content,
            },
        )

        return {"data": {"id": doc_id}}
    except Exception as e:
        print(e)
        return {"error": {"message": e.__str__()}}
