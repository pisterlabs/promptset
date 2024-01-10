import datetime
import random
from ai755 import settings
from ai75589674.models import UserReport, Black_token
from django.http import HttpResponse, HttpResponseBadRequest
import openai
import json


def generate_dental_report(
    predictions, token, email, media_path, tmedia_path=None
):
    try:
        gxl = []

        # Extract specific information from predictions and populate the report
        for prediction in predictions["predictions"]:
            # Make sure prediction is a dictionary and contains the expected keys
            if (
                not isinstance(prediction, dict)
                or "x" not in prediction
                or "width" not in prediction
            ):
               
                continue

            points = [
                (
                    prediction["x"] - prediction["width"] / 2,
                    prediction["y"] - prediction["height"] / 2,
                ),
                (
                    prediction["x"] + prediction["width"] / 2,
                    prediction["y"] - prediction["height"] / 2,
                ),
                (
                    prediction["x"] + prediction["width"] / 2,
                    prediction["y"] + prediction["height"] / 2,
                ),
                (
                    prediction["x"] - prediction["width"] / 2,
                    prediction["y"] + prediction["height"] / 2,
                ),
            ]
            gxl.append(prediction["class"])

            report_template = f"""Give me a dental radiology report about pathologies found close to {gxl}: {points}\n, 
            Note: 
            1. Don't use Numbers coordinates.
            2. determine treatment depending on caries with pulp approximation or overlapping.
            3. add treatment options for eaach case.
            4. Just USE A General FORMAT TO CREATE DENTAL REPORT
            5. Please use paragraph format
            """

        openai.api_key = settings.OPEN_API_KEY
        completion = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=report_template,
            max_tokens=1500,
            # Adjusting temperature=0.2
        )
        generated_report = completion.choices[0].text

        if UserReport.objects.filter(user_token__token1=token).exists():
            userobj = UserReport.objects.get(user_token__token1=token)
            userobj.report = generated_report,
            
            userobj.save()
            return None

        # Create the Black_token object
        ch = Black_token(token1=token)
        ch.save()  # Save Black_token

        # Create the UserReport object with the user_token field set
        ch_dbms = UserReport(email=email, report=generated_report, user_token=ch, media=media_path, tmedia=tmedia_path)
        ch_dbms.save()  # Save UserReport

    except Exception as e:
        print(e, "I'm calling from pdf generator")
        return HttpResponse("An error occurred while generating the report")
