!pip install honeyhive -q
import honeyhive
from honeyhive.sdk.utils import fill_template
from openai import OpenAI
import time

honeyhive.api_key = "HONEYHIVE_API_KEY"
client = OpenAI(api_key="OPENAI_API_KEY")

USER_TEMPLATE = "Write me an email on {{topic}} in a {{tone}} tone."
user_inputs = {
    "topic": "AI Services",
    "tone": "Friendly"
}
#"Write an email on AI Services in a Friendly tone."
user_message = fill_template(USER_TEMPLATE, user_inputs)

start = time.perf_counter()

openai_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=100,
    messages=[
      {"role": "system", "content": "You are a helpful assistant who writes emails."},
      {"role": "user", "content": user_message}
    ]
)

end = time.perf_counter()

request_latency = (end - start)*1000
generation = openai_response.choices[0].message.content
token_usage = openai_response.usage
response = honeyhive.generations.log(
    project="Sandbox - Email Writer",
    source="staging",
    model="gpt-3.5-turbo",
    hyperparameters={
        "temperature": 0.7,
        "max_tokens": 100,
    },
    prompt_template=USER_TEMPLATE,
    inputs=user_inputs,
    generation=generation,
    metadata={
        "session_id": session_id  # Optionally specify a session id to track related completions
    },
    usage=token_usage,
    latency=request_latency,
    user_properties={
        "user_device": "Macbook Pro",
        "user_Id": "92739527492",
        "user_country": "United States",
        "user_subscriptiontier": "Enterprise",
        "user_tenantID": "Acme Inc."
    }
)
from honeyhive.sdk.feedback import generation_feedback
generation_feedback(
    project="Sandbox - Email Writer",
    generation_id=response.generation_id,
    ground_truth="INSERT_GROUND_TRUTH_LABEL",
    feedback_json={
        "provided": True,
        "accepted": False,
        "edited": True
    }
)
import honeyhive

honeyhive.api_key = "HONEYHIVE_API_KEY"
honeyhive.openai_api_key = "OPENAI_API_KEY"

response = honeyhive.generations.generate(
    project="Sandbox - Email Writer",
    source="staging",
    input={
        "topic": "Model evaluation for companies using GPT-4",
        "tone": "friendly"
    },
)
