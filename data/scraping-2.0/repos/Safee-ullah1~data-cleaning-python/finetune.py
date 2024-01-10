from openai import OpenAI
client = OpenAI(
    api_key="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",
)

client.files.create(
    file=open("data.jsonl", "rb"),
    purpose="fine-tune",
)
