import json
import logging
import time
from datetime import datetime

import boto3
from anthropic import AI_PROMPT
from anthropic import HUMAN_PROMPT
from llm_bench_api.config import CloudConfig


logger = logging.getLogger(__name__)


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run BedRock inference and return metrics."""

    assert config.provider == "bedrock", "provider must be anthropic"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Set up connection
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

    # Generate
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    time_to_first_token = None
    output_tokens = 0
    times_between_tokens = []

    if "anthropic" in config.model_name:
        body = {
            "prompt": f"{HUMAN_PROMPT} {run_config['query']} {AI_PROMPT}",
            "max_tokens_to_sample": run_config["max_tokens"],
            "temperature": config.temperature,
        }
    elif "amazon" in config.model_name:
        body = {
            "inputText": "Human: Tell me a long history of WW2. \n\nAssistant:",
            "textGenerationConfig": {
                "maxTokenCount": run_config["max_tokens"],
                "temperature": config.temperature,
            },
        }
    else:
        raise ValueError(f"Unknown model name: {config.model_name}")

    if config.streaming:
        response = bedrock.invoke_model_with_response_stream(
            body=json.dumps(body),
            modelId=config.model_name,
        )
        stream = response.get("body")
        last_chunk = None
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    last_chunk = chunk
                    current_time = time.time()
                    if not first_token_received:
                        time_to_first_token = current_time - time_0
                        first_token_received = True
                    else:
                        assert previous_token_time is not None
                        times_between_tokens.append(current_time - previous_token_time)
                    previous_token_time = current_time

        if last_chunk:
            response_metrics = json.loads(last_chunk.get("bytes").decode()).get("amazon-bedrock-invocationMetrics", {})
            output_tokens = response_metrics.get("outputTokenCount")
    else:
        response = bedrock.invoke_model(
            body=json.dumps(body),
            modelId=config.model_name,
        )
        output_tokens = int(response["ResponseMetadata"]["HTTPHeaders"]["x-amzn-bedrock-output-token-count"])

    time_1 = time.time()
    generate_time = time_1 - time_0
    tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0

    metrics = {
        "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_tokens": run_config["max_tokens"],
        "output_tokens": output_tokens,
        "generate_time": generate_time,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token": time_to_first_token,
        "times_between_tokens": times_between_tokens,
    }

    return metrics
