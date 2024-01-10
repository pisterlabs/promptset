import openai
from langchain.llms import VLLMOpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
completion = openai.Completion.create(model="facebook/opt-125m",
                                      prompt="San Francisco is a")
print("Completion result:", completion)


llm = VLLMOpenAI(openai_api_key="EMPTY", openai_api_base="http://localhost:8000/v1", model="facebook/opt-125m")
print(llm.predict("San Francisco is a"))

"""
python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m

python test/vlla_openai_v1.py


curl http://localhost:8005/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
"""
