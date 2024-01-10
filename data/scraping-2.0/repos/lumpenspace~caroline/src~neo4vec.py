import openai
from neomodel import db, Property, FloatProperty, IntegerProperty, StringProperty, StructuredNode, StructuredRel, DateProperty

def get_embedding(content:str):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=content)
    return response

class FloatVectorProperty(Property):
    def __init__(self, length, *args, **kwargs):
        self.length = length
        self.vector = [FloatProperty() for _ in range(length)]
        super().__init__(*args, **kwargs)

    def inflate(self, value, obj):
        if len(value) != self.length:
            raise ValueError(f"Vector length must be {self.length}")
        return [prop.inflate(val, obj) for prop, val in zip(self.vector, value)]

    def deflate(self, value, obj):
        if len(value) != self.length:
            raise ValueError(f"Vector length must be {self.length}")
        return [prop.deflate(val, obj) for prop, val in zip(self.vector, value)]
    
class SimilarityRel(StructuredRel):
    similarity = FloatProperty(required=True)

class ContentSection(StructuredRel):
    index = IntegerProperty(required=True)

class ContentMixin():
    content = StringProperty(required=True)
    published_at = DateProperty()
    embedding = FloatVectorProperty(length=1536)

    def pre_save(self):
        response = get_embedding(self.content)
        self.embedding = response['data'][0]['embedding']

    @classmethod
    def get_similar(cls, content:str):
        return db.index.vector.queryNodes("content", 3, )
