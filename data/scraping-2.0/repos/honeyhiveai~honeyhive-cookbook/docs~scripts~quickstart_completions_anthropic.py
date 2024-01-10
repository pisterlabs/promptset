!pip install honeyhive -q
import honeyhive
from honeyhive.sdk.utils import fill_template
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import time

honeyhive.api_key = "HONEYHIVE_API_KEY"

anthropic = Anthropic(
    api_key="ANTHROPIC_API_KEY",
)

USER_TEMPLATE = f"{HUMAN_PROMPT} Write me an email on {{topic}} in a {{tone}} tone.{AI_PROMPT}"
user_inputs = {
    "topic": "AI Services",
    "tone": "Friendly"
}
#"Write an email on AI Services in a Friendly tone."
user_message = fill_template(USER_TEMPLATE, user_inputs)

start = time.perf_counter()

completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=300,
    prompt=user_message
)

end = time.perf_counter()

request_latency = (end - start)*1000
generation = completion.completion
token_usage = {
    "completion_tokens": anthropic.count_tokens(completion.completion),
    "prompt_tokens": anthropic.count_tokens(user_message),
    "total_tokens": anthropic.count_tokens(completion.completion) + anthropic.count_tokens(user_message)
}
response = honeyhive.generations.log(
    project="Sandbox - Email Writer",
    source="staging",
    model="claude-2",
    hyperparameters={
        "max_tokens_to_sample": 300,
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
