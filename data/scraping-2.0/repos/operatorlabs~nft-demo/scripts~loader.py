import json
import chromadb
import os
import openai
import dotenv

dotenv.load_dotenv()

class Summarizer:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.initial_message = {"role": "system", "content": "You are a helpful assistant."}

    def generate_summary(self, route_data):
        query_message = {
            "role": "user",
            "content": f"Describe what the following API route would be best used for: '{route_data}'"
        }

        response = openai.ChatCompletion.create(
            model=self.model,
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=[self.initial_message, query_message],
            temperature=0.8,  
            max_tokens=125  
        )

        return response.choices[0].message['content']

summarizer = Summarizer()

def handle_schema(data, schema):
    if '$ref' in schema:
        return handle_ref(data, schema['$ref'])
    elif 'allOf' in schema:
        all_schemas = {}
        for sub_schema in schema['allOf']:
            all_schemas.update(handle_schema(data, sub_schema))
        return all_schemas
    elif 'properties' in schema:
        properties = {}
        for key, value in schema['properties'].items():
            properties[key] = handle_schema(data, value)
        return properties
    elif 'enum' in schema:
        return schema
    else:
        return schema

def handle_ref(data, ref_string):
    schema_name = ref_string.replace('#/components/schemas/', '')
    schema_data = data['components']['schemas'].get(schema_name, {})

    schema_data = handle_schema(data, schema_data)

    return schema_data

def process_endpoint(data, endpoint, method, details):
    route_data = {'Endpoint': endpoint, 'Method': method.upper(), 'Description': details.get('summary', '')}

    parameters = details.get('parameters', [])
    if parameters:
        route_data['Parameters'] = [{param.get('name', ''): param.get('description', '')} for param in parameters]

    if 'requestBody' in details and 'content' in details['requestBody']:
        for content_type, schema in details['requestBody']['content'].items():
            processed_schema = handle_schema(data, schema['schema'])
            route_data['Request Body'] = {content_type: processed_schema}

    responses = details.get('responses', {})
    if responses:
        route_data['Responses'] = []
        for status, response in responses.items():
            resp_content = {'Status Code': status, 'Description': response.get('description', '')}
            if 'content' in response:
                for content_type, schema in response['content'].items():
                    processed_schema = handle_schema(data, schema['schema'])
                    resp_content['Content'] = {content_type: processed_schema}
            route_data['Responses'].append(resp_content)

    route_data['generated_description'] = summarizer.generate_summary(route_data)
    
    return route_data

def process_file(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        print(f"Successfully loaded JSON data from {filename}")
    except Exception as e:
        print(f"Error loading JSON data from {filename}: {e}")
        return

    filename_base = os.path.basename(filename)
    collection_name = filename_base.split('.')[0] 

    try:
        collection = chroma_client.create_collection(name=collection_name)
        print(f"Successfully created the {collection_name} collection")
    except Exception as e:
        print(f"Error creating the {collection_name} collection: {e}")
        return

    servers_data = data.get('servers', [])
    servers_dict = {server['url']: server.get('description', '') for server in servers_data}
    collection.modify(metadata=servers_dict)
    print(f"Successfully updated metadata for the {collection_name} collection")

    paths = data.get('paths', {})

    for endpoint, methods in paths.items():
        for method, details in methods.items():
            try:
                route_data = process_endpoint(data, endpoint, method, details)
                print(f"Successfully processed the {endpoint} endpoint with the {method} method")

                document = json.dumps(route_data)
                metadata = {"name": f"{endpoint}_{method.upper()}"}
                id = f"{endpoint}_{method.upper()}"

                collection.add(documents=[document], metadatas=[metadata], ids=[id])
                print(f"Successfully added document to the {collection_name} collection")
            except Exception as e:
                print(f"Error processing the {endpoint} endpoint with the {method} method: {e}")
                continue

def process_directory(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    for file in json_files:
        process_file(os.path.join(directory, file))

if __name__ == '__main__':
    chroma_client = chromadb.PersistentClient(path='../chroma.db')

    process_directory('../schemas')
