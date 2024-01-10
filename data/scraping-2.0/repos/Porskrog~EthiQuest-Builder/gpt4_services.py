import logging
import os
import time
from random import choice
from flask import jsonify
from dilemma_services import get_last_dilemma_and_option
from openai import OpenAI

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_CREATED = 201

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI API
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

######################################################################################################
# Utility Functions - getting dilemmas and options for user from GPT-4                               #
######################################################################################################

def call_gpt4_dilemma_api(full_prompt):
    try:
        # Record the time before the API call
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a leadership dilemma generator."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=250
        )
        # Process the response
        # generated_text = completion.choices[0].text
        generated_text = response.choices[0].message.content
        print(f"Generated text: {generated_text}")
        
        # Calculate and log the duration
        duration = time.time() - start_time
        logging.info(f"GPT-4 API call took {duration} seconds.")
        logging.info(f"Generated text: {generated_text}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 500
    
    logging.info(f"200 OK: Successfully called GPT-4 API.")
    # return generated_text
    return generated_text, 200  # Return the generated text and a success status code

def parse_gpt4_dilemma_response(generated_text):
    # Parsing logic to extract the dilemma, options, pros, and cons from the generated_text
    try:
        # Split the generated text into lines
        lines = generated_text.split("\n")

        # Initialize empty dictionaries to hold the parsed information
        dilemma = {}
        options = []
        option = {}

        # Loop over the lines and parse them
        for line in lines:
            if line.startswith("Context:"):
                dilemma['Context'] = line.split("Context:", 1)[1].strip()
            elif line.startswith("Description:"):
                dilemma['Description'] = line.split("Description:", 1)[1].strip()
            elif line.startswith("Option"):
                if option:  # If there is a previous option, add it to the list
                    options.append(option)
                option = {'text': line.split(":", 1)[1].strip()}
            elif line.startswith("- Pros:"):
                option['pros'] = line.split("Pros:", 1)[1].strip().split(", ")
            elif line.startswith("- Cons:"):
                option['cons'] = line.split("Cons:", 1)[1].strip().split(", ")

        if option:  # Add the last option if it exists
            options.append(option)

        # Validate that we have all the necessary components
        if 'Context' not in dilemma or 'Description' not in dilemma:
            raise Exception("Missing context, or description in the generated text.")
        if len(options) < 2:
            raise Exception("At least two options are required in the generated text.")

        # Now you can use `dilemma` and `options` as needed
    except Exception as e:
        logging.exception(f"Error while parsing the generated text: {e}")
        return jsonify({"status": "failure", "message": "Internal Server Error"}), 500

    logging.info(f"200 OK: Successfully parsed the dilemma from GPT-4 as dilemma and options. Dilemma: {dilemma}, Options: {options}")
    return dilemma, options

def generate_new_dilemma_with_gpt4(last_dilemma=None, last_option=None, user_id=None):
    if last_dilemma is None or last_option is None:
        if user_id is None:
            raise ValueError("If last_dilemma and last_option are not provided, user_id must be provided.")
        # Fetch them if they weren't provided (this assumes user.id is accessible from here)
        last_dilemma, last_option = get_last_dilemma_and_option(user_id)
  
    # Decide whether to make the new dilemma consequential of the former dilemma
    # make_consequential = choice([True, False])
    make_consequential = choice([True, True, False])

    # Generate the prompt for GPT-4
    # base_prompt = "You are a leadership dilemma generator.\n\n"  # System role instruction
    base_prompt = "Please generate an ethical and leadership dilemma related to program management with either two or three options. If the dilemma is more straightforward, provide two options and ONLY if it's more complex and warrants a third option, provide three options."
    if make_consequential and last_dilemma and last_option:
        full_prompt = f"{base_prompt} It must be a direct consequence of the previous dilemma which was: {last_dilemma.question} and the chosen option which was: {last_option.text}"
    else:
        full_prompt = base_prompt
    
    # Add the standard format to the prompt
    full_prompt += """
    Format the dilemma as follows:
    Context: {Comma separated important context characteristics as for example Public Sector, Private Sector, Fixed Price, Variable Costs, Fixed Scope, Waterfall, Agile, etc., max 10 words}
    Description: {Brief description, max 60 words}
    Option A: {Option A, max 20 words}
    - Pros: {Pros for Option A, max 20 words}
    - Cons: {Cons for Option A, max 20 words}
    Option B: {Option B, max 20 words}
    - Pros: {Pros for Option B, max 20 words}
    - Cons: {Cons for Option B, max 20 words}
    Option C: {Option C, if it exists, max 20 words}
    - Pros: {Pros for Option C, max 20 words}
    - Cons: {Cons for Option C, max 20 words}
    """

    logging.info(f"200 OK: Calling GPT-4 API to generate new dilemma. Prompt: {full_prompt}")
    # API call to GPT-4 to generate the new dilemma

    try:
        # API call to GPT-4 (assuming you have a function or method for this)
        generated_text, status_code = call_gpt4_dilemma_api(full_prompt)
        if status_code != 200:
            return jsonify({"status": "failure", "message": "Error in GPT-4 API call"}), status_code

        # Parsing logic (assuming you have a function or method for this)
        parsed_response = parse_gpt4_dilemma_response(generated_text)
        # parsed_response = parse_gpt4_dilemma_response(response)

        # Return the parsed response        
        logging.info(f"200 OK: Successfully generated a new dilemma with GPT-4: {parsed_response}")
        return parsed_response
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        return None

#######################################################################################################
# Utility Functions - getting contexts for user from GPT-4                                            #
#######################################################################################################

def call_gpt4_context_api(full_prompt):
    try:
        # Record the time before the API call
        start_time = time.time()
        response = client.completions.create(
            engine="davinci",
            prompt=full_prompt,
            max_tokens=250
        )
        # Process the response
        generated_text = response.choices[0].text
        
        # Calculate and log the duration
        duration = time.time() - start_time
        logging.info(f"GPT-4 API call took {duration} seconds.")
        logging.info(f"Generated text: {generated_text}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 500
    
    logging.info(f"200 OK: Successfully called GPT-4 API.")
    # return generated_text
    return generated_text, 200  # Return the generated text and a success status code

def parse_gpt4_context_response(generated_text):
    # Parsing logic to extract the context from the generated_text
    try:
        # Split the generated text into lines
        lines = generated_text.split("\n")

        # Initialize empty dictionaries to hold the parsed information
        context = {}

        # Loop over the lines and parse them
        for line in lines:
            if line.startswith("Context:"):
                context['Context'] = line.split("Context:", 1)[1].strip()

        # Validate that we have all the necessary components
        if 'Context' not in context:
            raise Exception("Missing context in the generated text.")

        # Now you can use `context` as needed
    except Exception as e:
        logging.exception(f"Error while parsing the generated text: {e}")
        return jsonify({"status": "failure", "message": "Internal Server Error"}), 500

    logging.info(f"200 OK: Successfully parsed the context from GPT-4 as context. Context: {context}")
    return context

def generate_new_context_with_gpt4():
    # Generate the prompt for GPT-4
    base_prompt = "Please generate a context for an ethical and leadership dilemma related to program management. The context should be related to the following characteristics: Public Sector, Private Sector, Fixed Price, Variable Costs, Fixed Scope, Waterfall, Agile, etc."
    full_prompt = base_prompt
    
    # Add the standard format to the prompt
    full_prompt += """
    Format the context as follows:
    Context: {Comma separated important context characteristics as for example Public Sector, Private Sector, Fixed Price, Variable Costs, Fixed Scope, Waterfall, Agile, etc., max 10 words}
    """

    logging.info(f"200 OK: Calling GPT-4 API to generate new context. Prompt: {full_prompt}")
    # API call to GPT-4 to generate the new context

    try:
        # API call to GPT-4 (assuming you have a function or method for this)
        generated_text, status_code = call_gpt4_context_api(full_prompt)
        if status_code != 200:
            return jsonify({"status": "failure", "message": "Error in GPT-4 API call"}), status_code

        # Parsing logic (assuming you have a function or method for this)
        parsed_response = parse_gpt4_context_response(generated_text)
        # parsed_response = parse_gpt4_context_response(response)

        # Return the parsed response        
        logging.info(f"200 OK: Successfully generated a new context with GPT-4: {parsed_response}")
        return parsed_response
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        return None