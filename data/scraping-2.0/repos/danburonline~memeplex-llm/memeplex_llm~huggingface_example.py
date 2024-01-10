"""Script to run prompt-based language model on HuggingFace Hub."""
import os
from dotenv import load_dotenv
from langchain.llms import HuggingFaceTextGenInference

load_dotenv()

llm = HuggingFaceTextGenInference(
    inference_server_url=os.environ.get("INFERENCE_ENDPOINT", "default_value"),
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    client=None,
    async_client=None,
)

print(
    llm(
        """Can you write a short introduction about the relevance
        of the term monopsony in economics?"""
    )
)
