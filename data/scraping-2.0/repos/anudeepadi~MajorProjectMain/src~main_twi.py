from fastapi import FastAPI, UploadFile, File
from fastapi import FastAPI, Request, Response, Form, HTTPException
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
import openai
import logging
import os

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.environ['OPENAI_API_KEY'] 

validator = RequestValidator(os.environ["TWILIO_AUTH_TOKEN"])

@app.post("/sms")
async def sms_webhook(request: Request, From: str = Form(...), Body: str = Form(...)):
    # Verify that the request came from Twilio
    form_ = await request.form()
    url = str(request.url).replace("http", "https")
    sig = request.headers.get("X-Twilio-Signature", "")

    if not validator.validate(
        url,
        form_,
        sig,
    ):
        logger.error("Error in Twilio Signature")
        raise HTTPException(
            status_code=403, detail="Error in Twilio Signature")

    # Get the incoming message and phone number from the request body
    message = Body
    phone_number = From

    logger.info(f"Message received from {phone_number}: {message}")

    # Send the message to the OpenAI API
    response_text = openai.Completion.create(
        engine="text-davinci-003",
        prompt=message,
        max_tokens=128,
        temperature=0.7,
    ).choices[0].text

    logger.info(f"OpenAI response: {response_text}")

    # Create a Twilio response object
    twiml_response = MessagingResponse()

    # Add the response text to the Twilio response
    twiml_response.message(response_text)

    # Return the Twilio response as the HTTP response
    return Response(content=str(twiml_response), media_type="application/xml")