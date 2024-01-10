import cohere
import os
from cohere.responses.classify import Example

# get cohere api key from .env
from dotenv import load_dotenv
import os

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

INTENTS = {'General QA': 0, 'Diagnose Brain Tumour': 1, 'Blood Work': 2}

BRAIN_TUMOUR = "Diagnose Brain Tumour"
OTHER = "Other"

def get_user_intent(user_message):

  examples = [
    Example("I need a tumour diagnoses on this brain scan.", BRAIN_TUMOUR),
    Example("Can you make a diagnoses for this brain MRI?", BRAIN_TUMOUR),
    Example("What is the cancer likelihood for this MRI scan of a patient's brain?", BRAIN_TUMOUR),
    Example("What is the probability of positive tumour diagnosis for this brain MRI.", BRAIN_TUMOUR),
    Example("I uploaded a brain scan, can you analyze and interpret it for me?", BRAIN_TUMOUR),
    Example("What is the survival rate for stage 2 lung cancer", OTHER),
    Example("What is the survival rate for brain tumour", OTHER),
    Example("How is indigestion cured?", OTHER),
    Example("What are the symptoms of diabetes?", OTHER),
  ]

  # Sends the classification request to the Cohere model
  user_intent = co.classify(
    model='large',
    inputs=[user_message],
    examples=examples
  )

  return user_intent.classifications[0].predictions
