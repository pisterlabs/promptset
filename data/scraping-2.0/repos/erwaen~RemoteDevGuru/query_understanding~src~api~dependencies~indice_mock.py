""""""
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity

class BaseIndice:
    def result(self):
        raise NotImplementedError()

class IndiceMaxLenException(Exception):
    """Exception raised for errors.

    Attributes:
        message -- explanation of the error
    """
    
    def __init__(self, salary, message="El numero de tokens del indice supero el maximo"):
        self.message = message
        super().__init__(self.message)

class IndiceMock(BaseIndice):

    def result(self, mes):
        resp = {"response": ["hola", " que ", "tal"]}
        if len(resp['response']) > 3:
            raise IndiceMaxLenException("la cantidad de respuesta del indice supero el limite")
        return resp
        # return {"response": self._get_coincidence_embedding(message=mes)}
    

    def _get_coincidence_embedding(self, message: str, result_number: int = 5) -> any:
        # Utilizamos pickle para mantener toda la metadata del archivo
        # beneficia, porque mantiene el tipo de dato  de las columnas originales
        # El tipo de dato en numpy 
        self.doc_dir = f"/usr/query_understanding/documentos/"
        datos: pd.DataFrame = pd.read_pickle(f'{self.doc_dir}emb.pk')
        
        busqueda_embed = get_embedding(message, engine="text-embedding-ada-002")
        datos["Similitud"] = datos['Embedding'].apply(lambda x: cosine_similarity(x, busqueda_embed))
        datos = datos.sort_values("Similitud", ascending=False)
        # return datos.iloc[:result_number][["texto", "Similitud", "Embedding"]]
        # print(datos.iloc[:result_number]["Similitud"])
        # if datos.iloc[:result_number]["Similitud"][0] < 0.9:
        #     pass
        return datos.iloc[:result_number]["texto"]
    
        # def embed_text(path="texto.csv"):
        #     conocimiento_df = pd.read_csv(path)
        #     conocimiento_df['Embedding'] = conocimiento_df['texto'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
        #     conocimiento_df.to_csv('embeddings.csv')
        #     return conocimiento_df