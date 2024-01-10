!pip install honeyhive -q
import honeyhive
import openai
import time

# import any other vector databases, APIs and other model providers you might need
honeyhive.api_key = "HONEYHIVE_API_KEY"
openai.api_key = "OPENAI_API_KEY"
honeyhive_eval = HoneyHiveEvaluation(
    project="Email Writer App",
    name="Max Tokens Comparison",
    description="Finding best max tokens for OpenAI chat models",
    dataset_name="Test",
    metrics=["Conciseness", "Number of Characters"] 
)
dataset = [
    {"topic": "Test", "tone": "test"},
    {"topic": "AI", "tone": "neutral"}
]

# in case you have a saved dataset in HoneyHive
from honeyhive.sdk.datasets import get_dataset
dataset = get_dataset("Email Writer Samples")
config =  {
      "name": "max_tokens_100",
      "model": "gpt-3.5-turbo",
      "provider": "openai",
      "hyperparameters": {
        "temperature": 0.5, 
        "max_tokens": 100
      },
      "chat": [
        {
            "role": "system",
            "content": "You are a helpful assistant who helps people write emails.",
        },
        {
            "role": "user",
            "content": "Topic: {{topic}}\n\nTone: {{tone}}."
        }
      ]
}
# parallelized version of the evaluation run code
import concurrent.futures

def parallel_task(data, config):
    data_run = []
    messages = honeyhive.utils.fill_chat_template(config["chat"], data)

    start = time.time()
    openai_response = openai.ChatCompletion.create(
        model=config["model"],
        messages=messages,
        **config["hyperparameters"]
    )
    end = time.time()

    honeyhive_eval.log_run(
        config=config, 
        input=data,
        completion=openai_response.choices[0].message.content,
        metrics={
            "cost": honeyhive.utils.calculate_openai_cost(
                config["model"], openai_response.usage
            ),
            "latency": (end - start) * 1000,
            "response_length": openai_response.usage["completion_tokens"],
            **openai_response["usage"]
        }
    )

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for data in dataset:
        for config in configs:
            futures.append(executor.submit(parallel_task, data, config))
    for future in concurrent.futures.as_completed(futures):
        # Do any further processing if required on the results
        pass
config =  {
      # same configuration as above here
      "hyperparameters": {"temperature": 0.5, "max_tokens": 400},
}

# identical Evaluation Run code as above
for data in dataset:
...
honeyhive_eval.log_comment("Results are decent")
honeyhive_eval.finish()
