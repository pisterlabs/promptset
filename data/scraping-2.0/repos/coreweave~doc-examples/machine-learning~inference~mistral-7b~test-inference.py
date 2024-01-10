import openai

# Replace <inference-url> with the URL shown by `kubectl get ksvc mistral-7b`.
openai.api_base = "<inference-url>" + "/v1"

# vLLM server is not authenticated.
openai.api_key = "none"

completion = openai.Completion.create(
    model="mistralai/Mistral-7B-v0.1",
    prompt="The mistral is",
    temperature=0.7,
    max_tokens=200, stop=".")

print(completion.to_dict_recursive())
