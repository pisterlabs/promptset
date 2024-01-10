""" Image Generation Module for AutoGPT."""
import io
import os
import sys
import uuid
import shutil
import subprocess
from base64 import b64decode
from pathlib import Path
import re

import openai
import requests
from PIL import Image

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.logs import logger
from autogpt.visionconfig import visionhack
from autogpt.commands.execute_code import execute_shell

CFG = Config()

@command("run_image", "Generate Image with stablediffusion", '"prompt": "<prompt>"')
def run_image(prompt: str, size: int = 768) -> str:
    """Generate an image with Stable Diffusion.

    Args:
        prompt (str): The prompt to use

    Returns:
        str: The filename of the image
    """
    current_dir = os.getcwd()
    # Change dir into workspace if necessary
    workspace_directory = f"{visionhack}/auto_gpt_workspace"
    if str(workspace_directory) not in current_dir:
        os.chdir(workspace_directory)

    # Execute the stablediffusion.py script with the provided prompt
    command_line = f'python stablediffusion.py --prompt "{prompt}"'
    print(f"Executing command 'run_image' in working directory...")

    result = subprocess.run(command_line, capture_output=True, shell=True, encoding="utf-8")
    output = result.stdout

    # Check for errors in the output
    if "Error" in output:
        return f"Error generating image: {output}"

    output_directory = f"{stablehome}/outputs/txt2img-samples/samples"
    workspace_directory = f"{visionhack}/images"

    # Find the highest numbered PNG file
    pattern = re.compile(r'^(\d{5})\.png$')
    max_number = -1
    filename = None

    for entry in os.listdir(output_directory):
        match = pattern.match(entry)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                filename = entry

    if filename:
        # Copy the file to the workspace directory
        generated_image_path = os.path.join(output_directory, filename)
        new_image_path = os.path.join(workspace_directory, filename)
        shutil.copyfile(generated_image_path, new_image_path)

        return f"generated image saved to {filename}"
    else:
        return "Error: Generated image not found."