# from langchain import basecomponent
# from ..src.bert import get_closest_feature, get_bert_embedding
#
# class BertComponent(basecomponent.BaseComponent):
#     def __init__(self, name):
#         super().__init__(name)
#
#     def process(self, data, embeddings):
#         user_embedding = get_bert_embedding(data)
#         return get_closest_feature(embeddings, user_embedding)