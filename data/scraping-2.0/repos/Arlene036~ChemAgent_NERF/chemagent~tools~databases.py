# import molbloom
import requests
from langchain.tools import BaseTool
from langchain.tools import StructuredTool
from langchain.embeddings import GPT4AllEmbeddings
from rdkit import Chem
import pandas as pd
import re
from chemagent.utils import *
import numpy as np
import pandas as pd
# import numba as nb
import json
import ast

import pandas as pd

def format_reaction_data(dataframe):
    formatted_str = ""

    for index, row in dataframe.iterrows():
        reactant_str = "reactants and reagents:" + row['reactants_smiles'] + "\n"
        product_str = "products:" + row['products_smiles'] + "\n"
        formatted_str += reactant_str + product_str + "\n"

    return formatted_str


def dataframe_to_json(df):
    data_dict = df.to_dict(orient='records')
    json_str = json.dumps(data_dict, indent=4, ensure_ascii=False)
    output_str = f"{json_str}"
    return output_str

# @nb.jit(nopython=True)
# def euclidean_distance_numba(vector1, vector2):
#     dist = 0.0
#     for i in range(len(vector1)):
#         dist += (vector1[i] - vector2[i]) ** 2
#     return np.sqrt(dist)

# def find_nearest_row_index(input_vector, dataframe, k = 3):
#     df_arrays = np.array([np.array(embedding) for embedding in dataframe['reactants_smiles_embedding']])
#     print('~'*20)
#     print(df_arrays.shape)
#     print('~'*20)
#     distances = np.apply_along_axis(euclidean_distance_numba, 1, df_arrays, input_vector)
#     nearest_row_indices = np.argsort(distances)[:k]
#     return nearest_row_indices

# def euclidean_distance(v1, v2):
#     return np.sqrt(np.sum((v1 - v2) ** 2))

def find_nearest_row_index(target_vector,dataframe, k):
    # 将字符串形式的向量转换为实际的Python列表
    dataframe['reactants_smiles_embedding'] = dataframe['reactants_smiles_embedding'].apply(lambda x: ast.literal_eval(x.replace('"','')))
    
    # 将target_vector与每个向量计算欧氏距离
    dataframe['distance'] = dataframe['reactants_smiles_embedding'].apply(lambda x: euclidean_distance(np.array(x), np.array(target_vector)))
    
    # 按照距离排序，并找到最近的k行的索引
    nearest_k_indices = dataframe.nsmallest(k, 'distance').index

    print('-'*20)
    print(nearest_k_indices)
    print('-'*20)
    return nearest_k_indices

def strip_non_alphanumeric(input_string):
    # 使用正则表达式匹配头尾除了`.`、空格和英文字母以外的字符，并替换为空字符串
    result_string = re.sub(r"^[^a-zA-Z. ]+|[^a-zA-Z. ]+$", "", input_string)
    return result_string

# class QueryUSPTOWithType(BaseTool):
#     name = "QueryUSPTOWithType"
#     description = "Input SMILES of reactants and possible reaction type, returns a file of similar reaction with reactants_smiles and products_smiles. You need to find the pattern of the chemical reaction from these similar reactionsfor final prediction of products. \
#         An example of input of this tool is \"C1=CC=C(C=C1)C2=CC=CC=C2, heteroatom alkylation and arylation\". "

#     url: str = None
#     data: pd.DataFrame = None
    

#     def __init__(
#         self,
#     ):
#         super(QueryUSPTOWithType, self).__init__()
#         self.data = pd.read_csv("../data/uspto_50k_embeded.csv")

#     def _run(self, smiles_reaction_type: str) -> str:
#     # def _run(self,  reaction_type: str) -> str:
#         """This function queries the reactants smiles as well as possible reaction type (separate by ,), returns a json file of similar reaction with reactants_smiles and products_smiles"""
#         """Reaction type can only be one of the ten types: heteroatom alkylation and arylation, acylation and related processes, C-C bond formation, heterocycle formation, protections, deprotections, reductions, oxidations, functional group interconversion, functional group addition (FGA) """
#         """Useful for reaction prediction, because similar reaction dataset is useful."""
#         smiles_reaction_type = strip_non_alphanumeric(smiles_reaction_type)
#         try:
#             smiles, reaction_type = smiles_reaction_type.split(", ")
#         except:
#             try:
#                 smiles, reaction_type = smiles_reaction_type.split(",")
#             except:
#                 return "Invalid input, please input SMILES (multiple molecule SMILES are separated by .) and reaction type (SMILES and reaction type are separated by ,)"
        
#         if reaction_type not in ["heteroatom alkylation and arylation", "acylation and related processes", "C-C bond formation", "heterocycle formation", "protections, deprotections", "reductions", "oxidations", "functional group interconversion", "functional group addition (FGA)"]:
#             return "Invalid input, please input one of the reaction types in the following list: heteroatom alkylation and arylation, acylation and related processes, C-C bond formation, heterocycle formation, protections, deprotections, reductions, oxidations, functional group interconversion, functional group addition (FGA)"
        
#         data_ = self.data[self.data["reaction_type"] == reaction_type]
#         if len(data_) == 0:
#             data_ = self.data.sample(10000)
#         target_vector = GPT4AllEmbeddings().embed_query(smiles) # √
#         row_indices = find_nearest_row_index(target_vector, data_, k = 5) # ×

#         # return dataframe_to_json(data_.iloc[row_indices][["reactants_smiles","products_smiles"]])
#         # return 0

#         # return '******'.join(['valid input',format_reaction_data(data_.sample(3)[['reactants_smiles','products_smiles']])])
#         return '******'.join(['valid input',format_reaction_data(self.data.iloc[row_indices][["reactants_smiles","products_smiles"]])])



#     async def _arun(self, query: str) -> str:
#         """Use the tool asynchronously."""
#         raise NotImplementedError()


class Query2SMILES(BaseTool):
    name = "Name2SMILES"
    description = "Input a molecule name, returns SMILES."
    url: str = None

    def __init__(
        self,
    ):
        super(Query2SMILES, self).__init__()
        self.url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"

    def _run(self, query: str) -> str:
        """This function queries the given molecule name and returns a SMILES string from the record"""
        """Useful to get the SMILES string of one molecule by searching the name of a molecule. Only query with one specific name."""

        # query the PubChem database
        r = requests.get(self.url.format(query, "property/IsomericSMILES/JSON"))
        # convert the response to a json object
        data = r.json()
        # return the SMILES string
        try:
            smi = data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
        except KeyError:
            return "Could not find a molecule matching the text. One possible cause is that the input is incorrect, input one molecule at a time."
        # remove salts
        return Chem.CanonSmiles(largest_mol(smi))

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class Query2CAS(BaseTool):
    name = "Mol2CAS"
    description = "Input molecule (name or SMILES), returns CAS number."
    url_cid: str = None
    url_data: str = None

    def __init__(
        self,
    ):
        super(Query2CAS, self).__init__()
        self.url_cid = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/{}/cids/JSON"
        )
        self.url_data = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
        )

    def _run(self, query: str) -> str:
        try:
            mode = "name"
            if is_smiles(query):
                mode = "smiles"
            url_cid = self.url_cid.format(mode, query)
            cid = requests.get(url_cid).json()["IdentifierList"]["CID"][0]
            url_data = self.url_data.format(cid)
            data = requests.get(url_data).json()
        except (requests.exceptions.RequestException, KeyError):
            return "Invalid molecule input, no Pubchem entry"

        try:
            for section in data["Record"]["Section"]:
                if section.get("TOCHeading") == "Names and Identifiers":
                    for subsection in section["Section"]:
                        if subsection.get("TOCHeading") == "Other Identifiers":
                            for subsubsection in subsection["Section"]:
                                if subsubsection.get("TOCHeading") == "CAS":
                                    return subsubsection["Information"][0]["Value"][
                                        "StringWithMarkup"
                                    ][0]["String"]
        except KeyError:
            return "Invalid molecule input, no Pubchem entry"

        return "CAS number not found"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


# class PatentCheck(BaseTool):
#     name = "PatentCheck"
#     description = "Input SMILES, returns if molecule is patented"

#     def _run(self, smiles: str) -> str:
#         """Checks if compound is patented. Give this tool only one SMILES string"""
#         try:
#             r = molbloom.buy(smiles, canonicalize=True, catalog="surechembl")
#         except:
#             return "Invalid SMILES string"
#         if r:
#             return "Patented"
#         else:
#             return "Novel"

#     async def _arun(self, query: str) -> str:
#         """Use the tool asynchronously."""
#         raise NotImplementedError()
