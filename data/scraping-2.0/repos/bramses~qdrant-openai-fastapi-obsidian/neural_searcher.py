# Import client library
import urllib.parse
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, FilterSelector, Batch
from qdrant_client import QdrantClient
import os
import openai
import logging
import uuid
from dotenv import load_dotenv
load_dotenv()


openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)


class NeuralSearcher:

    qdrant_client = None
    file_directory = None

    def __init__(self, collection_name, filenames=[], DELETE_FILES_FLAG=True):
        logging.info("Initializing NeuralSearcher")
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host='localhost', port=6333)
        self.file_directory = "data"
        self.filenames = filenames
        self.DELETE_FILES_FLAG = DELETE_FILES_FLAG

    def get_collection_info(self):
        logging.info("Getting collection info for {collection_name}".format(
            collection_name=self.collection_name))
        return self.qdrant_client.http.collections_api.get_collection(self.collection_name)

    def recreate_collection_from_scratch(self):
        logging.info("Recreating collection from scratch -- {collection_name}".format(
            collection_name=self.collection_name))
        create_collection(collection_name=self.collection_name,
                          qdrant_client=self.qdrant_client)

    def create_collection(self, collection_name):
        logging.info("Creating collection -- {collection_name}".format(
            collection_name=collection_name))
        create_collection(collection_name=self.collection_name,
                          qdrant_client=self.qdrant_client)

    def delete_points_by_filename(self, filenames):
        logging.info("Deleting points by filename -- {collection_name}".format(
            collection_name=self.collection_name))
        for filename in filenames:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="filename",
                                match=MatchValue(
                                    value="{filename}".format(filename=filename)),
                            ),
                        ],
                    )
                ),
            )

    def dry_run(self, filenames):
        logging.info("Dry run -- {collection_name}".format(
            collection_name=self.collection_name))
        to_add, to_delete = self.file_comparison(filenames=filenames)
        logging.info("Files to add: {to_add}".format(to_add=to_add))
        logging.info("Files to delete: {to_delete}".format(
            to_delete=to_delete))
        
        return to_add, to_delete

    def upload_filenames(self, filenames):
        logging.info("Uploading filenames to {collection_name}".format(
            collection_name=self.collection_name))

        to_add, to_delete = self.file_comparison(filenames=filenames)
        logging.info("Files to add: {to_add}".format(to_add=to_add))
        logging.info("Files to delete: {to_delete}".format(
            to_delete=to_delete))

        if len(to_add) > 0:
            # batch into 100 size chunks
            logging.info("Adding {len} files".format(len=len(to_add)))
            for i in range(0, len(to_add), 100):
                batch = to_add[i:i + 100]
                embeddings = create_embeddings(batch)
                vectors = map(lambda x: x["embedding"], embeddings)
                vectors = list(vectors)
                payload = map(lambda x: {"filename": x}, batch)
                upload_data(collection_name=self.collection_name, vectors=vectors,
                            payload=payload, qdrant_client=self.qdrant_client)

        if len(to_delete) > 0 and self.DELETE_FILES_FLAG:
            self.delete_points_by_filename(filenames=to_delete)

    def search(self, query, top=3):
        return search(collection_name=self.collection_name, query=query, top=top, qdrant_client=self.qdrant_client)

    def scroll(self, filename):
        return self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(
                            value="{filename}".format(filename=filename))
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vector=False,
        )

    def get_all(self):
        logging.info("Getting all points -- {collection_name}".format(
            collection_name=self.collection_name))

        points = []
        next_page_offset = 0

        while True:
            res = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vector=False,
                offset=next_page_offset
            )

            points.append(res[0])
            next_page_offset = res[1]

            if res[1] == None:
                break

        flat_list = [item for sublist in points for item in sublist]

        logging.info("Got {len} points".format(len=len(flat_list)))

        return flat_list

    def file_comparison(self, filenames):
        logging.info("Comparing files in data folder for insertion and deletion -- {collection_name}".format(
            collection_name=self.collection_name))
        points = self.get_all()
        points_list = list(points)

        point_filenames = list(
            map(lambda x: x.payload["filename"], points_list))
        ins, delt = compare_lists(filenames, point_filenames)

        logging.info("Inserting {ins} files".format(ins=ins))
        logging.info("Deleting {delt} files".format(delt=delt))

        return ins, delt


def open_file_in_obsidian(vault, filename):
    logging.info(
        "Converting {filename} to Obsidian URL".format(filename=filename))
    url_encoded_filename = urllib.parse.quote(filename)
    return "obsidian://advanced-uri?vault={vault}&filepath={filename}".format(filename=url_encoded_filename, vault=vault)


# https://stackoverflow.com/questions/49273647/python-recursive-function-by-using-os-listdir
def recursive(dir, all_files=[]):
    files = os.listdir(dir)
    for obj in files:

        if os.path.isfile(os.path.join(dir, obj)):
            logging.debug("File : "+os.path.join(dir, obj))
            all_files.append(obj)
        elif os.path.isdir(os.path.join(dir, obj)):
            logging.debug('called on dir: ', os.path.join(dir, obj))
            recursive(os.path.join(dir, obj), all_files)
        else:
            logging.error('Not a directory or file %s' %
                          (os.path.join(dir, obj)))

    return all_files


def compare_lists(filenames, points):
    to_insert = []
    to_delete = []

    logging.info("Comparing {filenames} and {points}".format(
        filenames=len(filenames), points=len(points)))

    for filename in filenames:
        if filename not in points:
            to_insert.append(filename)

    for point in points:
        if point not in filenames:
            to_delete.append(point)

    logging.info("Inserted: ", to_insert)
    logging.info("Deleted: ", to_delete)

    return to_insert, to_delete


def create_embeddings(query, model="text-search-davinci-doc-001"):
    logging.info("Creating embeddings for {query} with model = {model}".format(
        query=query, model=model))
    response = openai.Embedding.create(
        model=model,
        input=query
    )

    if not response.data:
        logging.error("No data returned from OpenAI Embedding API")
        return
    return response.data


def create_collection(collection_name, qdrant_client=None):
    logging.info("Creating collection {collection_name}".format(
        collection_name=collection_name))
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vector_size=12288,
        distance="Cosine"
    )


def upload_data(collection_name, vectors, payload, qdrant_client=None):
    logging.info("Uploading data to {collection_name}".format(
        collection_name=collection_name))

    qdrant_client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payload,
        # Vector ids will be assigned automatically
        ids=[str(uuid.uuid4()) for v in range(len(vectors))],
        batch_size=100  # How many vectors will be uploaded in a single request?
    )


def search(collection_name, query, top=10, qdrant_client=None):

    embeddings = create_embeddings(
        query, model="text-search-davinci-query-001")
    vectors = map(lambda x: x["embedding"], embeddings)
    vectors = list(vectors)
    vector = vectors[0]

    # Use `vector` for search for closest vectors in the collection
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=vector,
        query_filter=None,  # We don't want any filters for now
        limit=top  # 5 the most closest results is enough
    )
    # `search_result` contains found vector ids with similarity scores along with the stored payload
    # In this function we are interested in payload only
    payloads = [hit.payload for hit in search_result]
    return payloads


'''
TODO:
- keep a stable collection of data 
- upsert any filenames that are not in the collection [x]
   - go over each filename in the directory and skip if it's already in the collection
   - if not, create an embedding and upload it to the collection
   - if something is deleted, remove it from the collection
- host the service on a server
- create an ios shortcut that calls the service


embeddings = create_embeddings(filenames)

        vectors = map(lambda x: x["embedding"], embeddings)
        vectors = list(vectors)
        payload = map(lambda x: {"filename": x}, filenames)

        upload_data(collection_name=self.collection_name, vectors=vectors,
                    payload=payload, qdrant_client=self.qdrant_client)

'''
