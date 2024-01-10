import openai
import os
import pkg_resources
import logging

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY in the environment variables.")
openai.api_key = API_KEY

SYSTEM_MESSAGE = """
You are equipped with the capability to comprehend directory structure instructions given in plain text. Based on the text provided, you must determine the hierarchical directory structure and generate the appropriate structure format. Use the established standards to delineate directories and files, ensuring clarity in the representation.
"""

def get_structure_representation(prompt):
    """Fetch the directory structure representation from OpenAI API based on the provided instruction."""
    logging.debug("Sending prompt to OpenAI API for processing.")
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
    )
    
    logging.info("Received response from OpenAI API.")
    return response.choices[0]["message"]["content"]

def generate_directory_structure(plain_text_tree):
    """
    Generate a directory structure based on the given instruction using the packaged template and the OpenAI API.
    
    Args:
    - plain_text_tree (str): The plain text tree for the directory structure.
    
    Returns:
    - str: A string representing the generated directory structure.
    """
    
    # Access the packaged uniformatter.ppt using pkg_resources
    template_path = pkg_resources.resource_filename('dirbuilder', 'uniformatter.ppt')
    
    with open(template_path, "r") as file:
        template = file.read()
    
    # Embed the input instruction into the template
    prompt = template.replace("<DIRECTORY_STRUCTURE_TEXT_DUMP>", plain_text_tree)
    
    directory_representation = get_structure_representation(prompt)
    
    # Compare the result with expected based on mock tests and log the discrepancies
    # Only works when in DEBUG mode, so as not to interfere with the actual application
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        if plain_text_tree.strip() != directory_representation:
            logging.debug(f"Expected structure:\n{plain_text_tree}\nActual structure:\n{directory_representation}")

    return directory_representation
