import os
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger
import subprocess
import os
import tempfile
import json
from typing import Tuple

MAX_BATCH_SIZE = 2048  # Maximum number of texts allowed in a batch

tools: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "verify_lean_proof",
            "description": "Verify the correctness of a Lean proof",
            "parameters": {
                "type": "object",
                "properties": {
                    "proof": {
                        "type": "string",
                        "description": "The content of the Lean proof to verify",
                        "maxLength": 10000,
                    }
                },
                "required": ["proof"],
            },
        },
    }
]
initial_instruction: str = """
You will be provided with a proof request, and your task is to 
    (1) write lean code for that proof, 
    (2) return a the code in a JSON format for 'function calling', 
    (3) interpret the response from the API and iterate if necessary. 
You are dedicated and will keep trying until you achieve success. 
Once you get a successful response, please summarize it for the user, including the lean code and a brief explainer.
For example, if the user asks for a proof that 1+1=2,
you might respond with a tool call for `{"proof":"example : 1 + 1 = 2 := rfl"}`
"""
chat_model: str = "gpt-4-1106-preview"
temp: float = 0.7
seed: int = 42
verbose: bool = True
max_successive_tries: int = 5
G = "\033[32m"  # green

MAX_PROOF_SIZE = 10_000
TEMP_FILE_PLACEHOLDER = "proof"


def verify_lean_proof(proof: str) -> Tuple[str, int]:
    # Input validation
    if not isinstance(proof, str):
        return json.dumps({"status": "error", "message": "Proof must be a string"}), 400

    if len(proof) > MAX_PROOF_SIZE:
        return json.dumps({"status": "error", "message": "Proof is too large"}), 400

    with tempfile.NamedTemporaryFile(
        suffix=".lean", mode="w", delete=False
    ) as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(proof)

    try:
        result = subprocess.run(
            ["lean", temp_file_name], capture_output=True, text=True, timeout=30
        )
        output = result.stdout.replace(temp_file_name, TEMP_FILE_PLACEHOLDER)
        error = result.stderr.replace(temp_file_name, TEMP_FILE_PLACEHOLDER)

        if result.returncode == 0:
            return (
                json.dumps(
                    {"status": "success", "message": "Proof is valid", "output": output}
                ),
                200,
            )
        else:
            return (
                json.dumps(
                    {
                        "status": "error",
                        "message": "Proof is invalid",
                        "output": output,
                        "error": error,
                    }
                ),
                400,
            )

    except subprocess.TimeoutExpired:
        return json.dumps({"status": "error", "message": "Verification timed out"}), 400
    except Exception as e:
        sanitized_message = str(e).replace(temp_file_name, TEMP_FILE_PLACEHOLDER)
        return json.dumps({"status": "error", "message": sanitized_message}), 500
    finally:
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)


def main():
    # Get the credentials:
    load_dotenv()
    api_key: str = os.environ.get("LLM_API_KEY")

    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    logger.info(f"Client initialized with API key ...{api_key[-6:]}\n")

    # Configure the initial interaction
    # Get a prompt from the user
    time.sleep(0.1)
    prompt: str = input("Enter a prompt: ")
    if verbose:
        logger.info(f"(Submitting request from user to {chat_model})")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": initial_instruction},
        {"role": "user", "content": prompt},
    ]

    break_tool_loop: bool = False
    tries: int = 0
    while not break_tool_loop:

        if tries >= max_successive_tries:
            logger.critical(
                f"Could not get working code after {tries} tries; breaking out of loop"
            )
            break_tool_loop = True

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=chat_model,
            seed=seed,
            temperature=temp,
            tools=tools,
        )

        finish_reason: str = chat_completion.choices[0].finish_reason
        if finish_reason == "tool_calls":
            tool_call_id: str = chat_completion.choices[0].message.tool_calls[0].id
            call_string: str = (
                chat_completion.choices[0].message.tool_calls[0].function.arguments
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": call_string,
                    "tool_call_id": tool_call_id,
                    "name": "verify_lean_proof",
                }
            )
            if verbose:
                logger.info(
                    f"{chat_model} requested tool call (ID:{tool_call_id}) {call_string}"
                )
                logger.info("(Submitting code from GPT to lean proof assistant)")
            proof_text: str = json.loads(call_string)["proof"]
            proof_response, proof_code = verify_lean_proof(proof=proof_text)
            if verbose:
                logger.info(
                    f"Response {proof_code} from proof assistant: {proof_response}"
                )
                logger.info(
                    "(Submitting response from lean proof assistant back to GPT for review)"
                )

            messages.append(
                {
                    "role": "function",
                    "content": proof_response,
                    "name": "verify_lean_proof",
                    "tool_call_id": tool_call_id,
                }
            )
            tries += 1
        elif finish_reason == "stop":
            message_content: str = chat_completion.choices[0].message.content
            messages += [{"role": "assistant", "content": message_content}]
            if verbose:
                logger.info(f"\nAssistant: {message_content}")  # extra logging
            else:
                print(G + f"Assistant: {message_content}")
            tries = 0  # reset counter
            break_tool_loop = True


if __name__ == "__main__":
    main()
