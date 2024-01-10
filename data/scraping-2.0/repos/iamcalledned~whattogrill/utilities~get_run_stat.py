from openai import OpenAI
client = OpenAI()

run = client.beta.threads.runs.retrieve(
  thread_id="thread_abc123",
  run_id="thread_A9DaL1y0rLdqxNhw6lps58Bg"
)
print(run)
