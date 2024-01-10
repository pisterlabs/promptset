from openai import OpenAI
client = OpenAI()

run = client.beta.threads.runs.retrieve(
  thread_id="thread_abc123",
  run_id="run_abc123"
)
print(run)



"""Retrieve runBeta

get https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}

Retrieves a run.
Path parameters
thread_id
string
Required

The ID of the thread that was run.
run_id
string
Required

The ID of the run to retrieve.
Returns

The run object matching the specified ID."""



"""{
  "id": "run_abc123",
  "object": "thread.run",
  "created_at": 1699075072,
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "status": "completed",
  "started_at": 1699075072,
  "expires_at": null,
  "cancelled_at": null,
  "failed_at": null,
  "completed_at": 1699075073,
  "last_error": null,
  "model": "gpt-3.5-turbo",
  "instructions": null,
  "tools": [
    {
      "type": "code_interpreter"
    }
  ],
  "file_ids": [
    "file-abc123",
    "file-abc456"
  ],
  "metadata": {}
}
"""