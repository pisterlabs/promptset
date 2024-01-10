import requests
import os
import openai
import logging

# Set up logging
logging.basicConfig(filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def generate_response(email_data, model, max_tokens, prompt_template, local_server, system_prompt, positive_response):
    prompt = prompt_template.format(email_body=email_data['Body'])

    try:
        if os.getenv('USE_LOCAL', 'false').lower() == 'true':
            # Send a POST request to the local server
            response = requests.post(
                local_server,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            # Extract the generated response
            response_content = response.json()["choices"][0]["message"]["content"]
        else:
            # Use OpenAI API
            openai.api_key = os.getenv('OPENAI_API_KEY')
            response = openai.ChatCompletion.create(
              model=model,
              messages=[
                  {"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}
              ]
            )
            # Extract the generated response
            response_content = response['choices'][0]['message']['content']

        # Prepare the response data
        is_junk = response_content.strip().startswith(positive_response)

        return is_junk
    except Exception as e:
        logging.error(f'An error occurred: {e}')
        return None
