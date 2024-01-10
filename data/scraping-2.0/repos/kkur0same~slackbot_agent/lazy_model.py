from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings, HuggingFaceInstructEmbeddings

''' lazy model and embeddings loading'''

class LazyModel:
    def __init__(self, config):
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # load model here based on self.config
            if self.config['type'] == 'ChatOpenAI':
                #self._model = ChatOpenAI(temperature=self.config['temperature'], model_name=self.config['model_name'])
                model_kwargs = self.config.copy()
                model_kwargs.pop('type', None)
                model_kwargs.pop('name', None)
                self._model = ChatOpenAI(**model_kwargs)
        return self._model


class LazyEmbedding:
    def __init__(self, config):
        self.config = config
        self._embedding = None

    @property
    def embedding(self):
        if self._embedding is None:
            if self.config['type'] == 'HuggingFaceInstructEmbeddings':
                self._embedding = HuggingFaceInstructEmbeddings(
                    embed_instruction=self.config.get('embed_instruction'),
                    query_instruction=self.config.get('query_instruction')
                )
            elif self.config['type'] == 'OpenAIEmbeddings':
                self._embedding = OpenAIEmbeddings()
            elif self.config['type'] == 'HuggingFaceEmbeddings':
                model_name = self.config.get('model_name')
                self._embedding = HuggingFaceEmbeddings(model_name=model_name)
        return self._embedding
