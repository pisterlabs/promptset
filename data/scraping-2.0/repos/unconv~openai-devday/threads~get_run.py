import openai

run = openai.beta.threads.runs.retrieve(
    run_id="",
    thread_id="",
)

print(run)