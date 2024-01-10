from openai import OpenAI
client = OpenAI()

runs = client.beta.threads.runs.list(
  "ththread_A9DaL1y0rLdqxNhw6lps58Bg"
)
print(runs)

