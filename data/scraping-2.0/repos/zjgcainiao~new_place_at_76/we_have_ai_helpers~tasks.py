#  this one should be in the tasks.py file.
from celery import shared_task
from openai import OpenAI
from we_have_ai_helpers.models import OpenAIModel
from django.utils import timezone
import datetime
from automanshop.celery import app
from django.db import transaction

# @app.task


def get_openai_model_list():

    with transaction.atomic():
        # Your code here
        client = OpenAI()

        response = client.models.list()

        for model_data in response.data:
            model, created = OpenAIModel.objects.get_or_create(
                model_id=model_data.id,
                defaults={
                    'model_created': timezone.make_aware(datetime.datetime.fromtimestamp(model_data.created)),
                    'model_owned_by': model_data.owned_by
                }
            )


# tasks.py
@shared_task
def add(x, y):
    return x + y
