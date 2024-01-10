"""
Class for generating consensus based on agent voting.

Similar to self-consistency but considering agent negatiations.
"""

# langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

import pandas as pd
from pandas import DataFrame
import pdb
import json
from typing import Any, Mapping, Optional, Protocol, Dict, List

class Congress:
    def __init__(self):
        return

    def read_laws(
        self, 
        jsonl_path: str, 
        label_map: Optional[Dict[str, str]] = None, 
        select_cols: Optional[List] = None
    ) -> DataFrame:
        """Return a DataFrame object, which contains training data from which we can 
        generate mini-batches. 

        Args:
            jsonl_path: Path of the JSONL files. Each JSONL file contains laws with the 
                following information:
            
                url_description          https://www2.congreso.gob.pe/Sicr/TraDocEstPro...
                estado_ley                                            Publicado El Peruano
                titulo_ley                     REGL.CONGRESO 37/MODIFICA INCISO 4) DEL....
                objetivo_ley             Propone modificar el inciso 4) del artículo 37...
                codigo_ley                                                           00351
                periodo_parlamentario                                          2016 - 2021
                legislatura                             Primera Legislatura Ordinaria 2016
                fecha_presentacion                                              09/30/2016
                proponente                                                        Congreso
                grupo_parlamentario                                 Peruanos por el Kambio
                autores                  [De Belaunde De Cárdenas  Alberto, Bruce Monte...
                url_pdf                  http://www2.congreso.gob.pe/Sicr/TraDocEstProc...
                nombre_comision                                [Constitución y Reglamento]
                sesion                                                         2020 - 2021

            label_map: Dictionary of the labels allowed to be part of the training 
                dataset (positive and negative examples).

            select_cols:  List of strings to select some columns in the generated DataFrame. 

        Returns:
            A DataFrame containing valid laws to form a training dataset.
        """
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f]
        
        df = pd.DataFrame(lines)        
        if select_cols:
            df = df[select_cols]
        if label_map:
            df = df.query(f"estado_ley == {list(label_map)}")
        pdb.set_trace()
        return

    def run(self) -> str:
        jsonl_path = './data/peru/laws/crawled/September_2016.html.jsonl'
        label_map = {'Al Archivo': 'pass', 'Publicado El Peruano': 'pass'}
        select_cols = ['estado_ley', 'titulo_ley', 'objetivo_ley', 'codigo_ley', 'proponente', 'grupo_parlamentario']
        self.read_laws(jsonl_path, label_map, select_cols)
        return ""

if __name__ == '__main__':
    congress = Congress()
    congress.run()
    
    
