import os
from open_api_key import OPENAI_API_KEY
from transform_data import data_transformation
from superduperdb import superduper
from superduperdb.db.mongodb.query import Collection
from superduperdb.container.document import Document
from superduperdb.ext.numpy.array import array
from superduperdb.container.vector_index import VectorIndex
from superduperdb.container.listener import Listener
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
from superduperdb.ext.openai.model import OpenAIEmbedding
model = OpenAIEmbedding(model='text-embedding-ada-002')


mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")



def create_database(data):
    """
    Define the DB and the collection and store the data into the database
    """
    db =superduper(mongodb_uri)
    collection = Collection('adidas_search_engine')
    db.execute(collection.insert_many([Document(r) for r in data]))
    return db, collection

def search_functionality(data, model=model):
    """
    Extend the database  into vector search functionality by adding the embedding model and specifing the vector index keys
    """
    db, collection = create_database(data)
    db.add(
            VectorIndex(
                  identifier=f'adidas-index-{model.identifier}',
                  indexing_listener=Listener(
                  model=model,
                  key='description',
                  select=collection.find(),
                  predict_kwargs={'max_chunk_size': 1000},
                   ),
            compatible_listener=Listener(

                  model=model,
                  key='name',
                  select=collection.find(),
                  active=False,
               ),
           )
        )
    return db, collection, model




if __name__ == '__main__':
    file_path = 'adidas_usa.csv'
    data_dic = data_transformation(file_path)
    db, collection, model = search_functionality(data_dic)
    r = db.execute(collection.find_one())
    print(r.unpack())
    print(model.predict('This is a test', one=True))
    cur = db.execute(
    collection
        .like({'description': 'a comfortable shoes for every day errands','name': 'a sport shoe'}, n=10, vector_index=f'adidas-index-{model.identifier}')
        .find({}, {'name': 1, 'color': 1, 'images':1, 'description': 1})
    )

    for r in cur:
        print(r.unpack())