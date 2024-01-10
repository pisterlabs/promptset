# import the necessary packages
from ripple_predict.models import SocialMediaPost
from ripple_predict.models import Prediction
from ripple_predict.models import Embedding
from django.conf import settings
from django.template.loader import get_template
from celery import shared_task
import openai
import json
import os

# set the OpenAI API key
openai.api_key = settings.OPENAI_API_KEY


@shared_task
def classify_post_with_prompt(
        smp_id,
        experiment,
        prompt_filename,
        model="gpt-3.5-turbo"
):
    # grab the social media post from the database
    smp = SocialMediaPost.objects.get(id=smp_id)

    # build the prompt path
    template_path = os.path.join(
        "ripple_predict",
        prompt_filename
    )

    # construct the prompt
    template = get_template(template_path)
    prompt = template.render({
        "post": smp.text,
    }).strip()

    # submit the prompt to OpenAI and obtain the response, then parse out the
    # prediction from the JSON blob
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    blob = json.loads(completion.choices[0].message.content)
    predicted_label = blob["label"]

    # store the prediction in the database
    prediction = Prediction(
        smp=smp,
        experiment=experiment,
        response=completion,
        prediction=predicted_label
    )
    prediction.save()


@shared_task
def compute_embeddings(
        smp_id,
        experiment,
        model="text-embedding-ada-002"
):
    # grab the social media post from the database
    smp = SocialMediaPost.objects.get(id=smp_id)

    # submit the embedding request to OpenAI
    response = openai.Embedding.create(
        model=model,
        input=smp.text
    )
    vector = response["data"][0]["embedding"]

    # store the embedding in the database
    embedding = Embedding(
        smp=smp,
        experiment=experiment,
        response=response,
        embeddings=json.dumps(vector)
    )
    embedding.save()
