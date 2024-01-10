from superduperdb.ext.openai import OpenAIEmbedding

model = OpenAIEmbedding(identifier='text-embedding-ada-002')

model.predict(
    X='input_col',
    db=db,
    select=Collection(name='test_documents')
)
