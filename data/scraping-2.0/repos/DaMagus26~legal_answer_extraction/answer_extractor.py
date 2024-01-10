from legal_answer_extraction.vector_db.weaviate_db import WeaviateDB
from sentence_transformers import SentenceTransformer
from legal_answer_extraction.article_db.pandas_article_db import PandasArticleDB
from legal_answer_extraction.qa_model.openai_qa import OpenAIModel
from transformers import pipeline


class AnswerExtractor:

    def __init__(
            self,
            vector_db,
            article_db,
            qa_model,
            sequence_encoder,
    ):
        self.vector_db = vector_db
        self.article_db = article_db
        self.qa_model = qa_model
        self.gen_model = OpenAIModel('sk-uJPlEJyqRg8jeYD1rSRxzV0DyXHzQtNb', 'https://api.proxyapi.ru/openai/v1')
        self.sequence_encoder = sequence_encoder
        self.certainty_thresh = 0.7

    def find(self, query):
        enc_query = self.sequence_encoder.encode(query).tolist()
        vdb_response = self.vector_db.find(enc_query)

        contexts = [item['article_text'] for item in vdb_response]
        ids = [item['_additional']['id'] for item in vdb_response]

        results = []
        for ctx_id, ctx in zip(ids, contexts):
            results.append((ctx_id, self.qa_model(question=query, context=ctx)))

        final_result = list(filter(lambda x: x[1]['score'] >= self.certainty_thresh, results))

        prospect_articles = '\n'.join([text['article_name'] + '\n' + text['article_text'] for text in final_result])
        text_response = self.gen_model(query, prospect_articles)

        response = {
            'article_text': text_response
        }

        return response

