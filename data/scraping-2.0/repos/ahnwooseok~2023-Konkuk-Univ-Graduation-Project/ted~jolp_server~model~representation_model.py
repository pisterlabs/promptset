from openai import OpenAI
from bertopic.representation._base import BaseRepresentation
from tqdm import tqdm
import re

class RepresentationModel(BaseRepresentation):
    def __init__(self,
                model: str,
                delay_in_seconds: float = None,
                nr_docs: int = 5, # the number of documents passed to OpenAI
                diversity: float = None, 
                doc_length: int = 512,
                ):
        self.model = model
        self.delay_in_seconds = delay_in_seconds
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.prompt = '''I have a topic that contains the following documents: 
        [DOCUMENTS]
        The topic is described by the following keywords: [KEYWORDS]

        Based on the information above, please give a korean word that describes this topic the most in the following format:
        topic: <korean word>
        '''
        self.client = OpenAI()

    def extract_topics(self, topic_model, documents, c_tf_idf, topics):

        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf,
            documents,
            topics,
            500,
            self.nr_docs,
            self.diversity
        )

        updated_topics = {}

        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            
            keywords = ", ".join(list(zip(*topics[topic]))[0])

            truncated_docs = [self.truncate_document(topic_model, self.doc_length, doc) for doc in docs]

            prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))

            to_replace = ""
            
            for doc in truncated_docs:
                to_replace += f"- {doc}\n"
            prompt = prompt.replace("[DOCUMENTS]", to_replace)
            
            messages = [
                {"role":"system", "content": "You are a helpful assistant"},
                {"role":"user", "content": prompt}
            ]
            kwargs = {"model": self.model, "messages":messages}

            response = self.client.chat.completions.create(**kwargs)


            label = response.choices[0].message.content.strip().replace("topic: ", "")
            representation = re.sub('\(.*\)', '', label).strip()
            updated_topics[topic] = [(representation,1)]


        return updated_topics

    
    def truncate_document(self, topic_model, doc_length, document):
        tokenizer = topic_model.vectorizer_model.build_tokenizer()
        truncated_document = " ".join(tokenizer(document)[:doc_length])

        return truncated_document