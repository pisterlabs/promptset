import os, sys, json

import weaviate
import openai

openai.api_key = os.environ.get('OPENAI_API_KEY')
EMBEDDIING_MODEL = "text-embedding-ada-002"
DEFAULT_FIELD = "content"

class WeaviateWrapper:
    # Create decorator for normalizing class names, which are internally capitalized
    # This affects the query results, as the normalized name is returned in the results
    def _normalize_class_name(method):
        def wrapper(self, class_name, *args, **kwargs):
            class_name = class_name[0].upper() + class_name[1:]
            return method(self, class_name, *args, **kwargs)
        return wrapper
    
    def __init__(self) -> None:
        # Get the WEAVIATE_API_KEY from the environment
        weaviate_apikey = os.environ.get('WEAVIATE_API_KEY')
        # Get the WEAVIATE_ENDPOINT from the environment
        weaviate_endpoint = os.environ.get('WEAVIATE_ENDPOINT')
        # Get the OPENAI_API_KEY from the environment
        openai_apikey = os.environ.get('OPENAI_API_KEY')

        # Instantiate the client with the auth config
        auth_config = weaviate.AuthApiKey(api_key=weaviate_apikey)
        self.client = weaviate.Client(
            url=weaviate_endpoint,
            auth_client_secret=auth_config,
            additional_headers = {
                "X-OpenAI-Api-Key": openai_apikey,
            },
        )

        # Simple test to see if the client is connected
        try:
            self.client.get_meta()
        except weaviate.UnexpectedStatusCodeException as e:
            sys.stderr.write("Error connecting to Weaviate: {}".format(e))
            sys.exit(1)
    
    def _make_default_field_value(self, value):
        return {DEFAULT_FIELD: value}

    @_normalize_class_name
    def insert_chunk(self, class_name, chunk):
        data_object = {DEFAULT_FIELD: chunk}
        oai_resp = openai.Embedding.create(input = [chunk], model=EMBEDDIING_MODEL)
        uuid = self.client.data_object.create(
            data_object=data_object,
            class_name=class_name,
            vector=oai_resp['data'][0]['embedding'],
        )
        return uuid
    
    @_normalize_class_name
    def insert_md_document(self, class_name, document):
        chunks = []
        breadcrumbs = []
        # Break down paragraph into sentences with breadcrumbs
        for section in document.split("\n\n"):
            section = section.strip()
            if section.startswith("#"):
                # Check the headings level by counting the number of hashes in the beggining of the line
                heading_level = section.split(" ")[0].count("#")
                heading = " ".join(section.split(" ")[1:])
                while len(breadcrumbs) == 0 or len(breadcrumbs) != heading_level or breadcrumbs[-1] != heading:
                    # If not in the breadcrumbs, add it
                    if len(breadcrumbs) == heading_level - 1:
                        breadcrumbs.append(heading)
                    # If in the breadcrumbs, replace it
                    elif len(breadcrumbs) == heading_level:
                        breadcrumbs[-1] = heading
                    # If at a higher level, pop it
                    elif len(breadcrumbs) > heading_level:
                        breadcrumbs.pop()
                    # If at a lower level, append None
                    elif len(breadcrumbs) < heading_level - 1:
                        breadcrumbs.append(" * ")
            else:
                # If the section is not a heading, append it to the chunks, including the breadcrumbs
                chunks.append("{}\n\n{}".format(" > ".join(breadcrumbs), section))

        # Insert each chunk into Weaviate
        for chunk in chunks:
            self.insert_chunk(class_name, chunk)
        
    def _generate_embedding(self, text):
        oai_resp = openai.Embedding.create(input = [text], model=EMBEDDIING_MODEL)
        return oai_resp['data'][0]['embedding']
    
    @_normalize_class_name
    def run_near_query(self, class_name, query, k=10):
        vector = self._generate_embedding(query)
        result = self.client.query.get(class_name, [DEFAULT_FIELD]).with_near_vector({"vector": vector}).with_limit(k).do()
        return [x[DEFAULT_FIELD] for x in result['data']['Get'][class_name]]

    @_normalize_class_name
    def delete_class(self, class_name):
        self.client.schema.delete_class(class_name=class_name)
    
    def dump(self):
        result = self.client.data_object.get()
        return result["objects"]

# Sample usage:
# weaviate = WeaviateWrapper()
# with open("test/samples/sample_md.md", "r") as f:
#     sample_md = f.read()
# weaviate.insert_md_document("test_class", sample_md)
# weaviate.insert_line("test_class", "The hills are flat... the water is dry... who the heck talks like this?")
# weaviate.insert_line("test_class", "Sulfiric acid has many uses in metallurgy.")
# weaviate.insert_line("test_class", "The carpenters are in need of more 2x4s")
# print(weaviate.run_near_query("test_class", "skip", 1))
# weaviate.delete_class("Test_class")
# print(weaviate.dump())
