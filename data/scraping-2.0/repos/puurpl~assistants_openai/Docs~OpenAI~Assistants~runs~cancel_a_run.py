from openai import OpenAI
client = OpenAI()

run = client.beta.threads.runs.cancel(
  thread_id="thread_abc123",
  run_id="run_abc123"
)
print(run)



"""Cancel a runBeta

post https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}/cancel

Cancels a run that is in_progress.
Path parameters
thread_id
string
Required

The ID of the thread to which this run belongs.
run_id
string
Required

The ID of the run to cancel.
Returns

The modified run object matching the specified ID."""



"""{
  "id": "run_abc123",
  "object": "thread.run",
  "created_at": 1699076126,
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "status": "cancelling",
  "started_at": 1699076126,
  "expires_at": 1699076726,
  "cancelled_at": null,
  "failed_at": null,
  "completed_at": null,
  "last_error": null,
  "model": "gpt-4",
  "instructions": "You summarize books.",
  "tools": [
    {
      "type": "retrieval"
    }
  ],
  "file_ids": [],
  "metadata": {}
}
"""