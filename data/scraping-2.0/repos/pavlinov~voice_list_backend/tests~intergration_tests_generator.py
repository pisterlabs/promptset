import openai
import yaml
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

yaml_file_path = 'tests/integration-requests.yaml'

# Load the YAML data
with open(yaml_file_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

# Extract the integration data from the YAML
integration_data = yaml_data['integration']

model = 'gpt-3.5-turbo'

for integration_name, integration_content in integration_data.items():
    request_data = integration_content['request']
    response_data = integration_content['response']

    # Define the prompt
    prompt = f"""
    Curl Request:
    {request_data}

    Response:
    {response_data}
    
    Make integration test, using "pytest", for {integration_name} api.
    """

    print(prompt)

    response = openai.Completion.create(
        prompt=prompt,
        max_tokens=2048,
        temperature=0,
        n=1,
        stop=None,
        model="text-davinci-003", # +
    )

    print("response"*10)
    print(response)

    # Extract the generated code from the response
    generated_code = response.choices[0].text.strip()

    integration_name = integration_name.replace('-', '_')

    # Define the file path to save the generated code
    output_file_path = f'tests/integration/output/test_{integration_name}.py'

    # Save the generated code to a file
    with open(output_file_path, 'w') as file:
        file.write(generated_code)

    print(f"Generated code for {integration_name} saved to: {output_file_path}")
