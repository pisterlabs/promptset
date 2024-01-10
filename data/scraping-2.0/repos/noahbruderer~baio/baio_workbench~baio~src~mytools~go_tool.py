import os
import mygene
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any
from typing import Optional
import concurrent.futures
from langchain.tools import tool
from src.non_llm_tools.utilities import Utils
from src.mytools.key_word_extractor_tool import NaturalLanguageExtractors

class GeneGOExtractor:
    """A class for extracting Gene Ontology (GO) information for a specific gene."""

    def __init__(self, gene_id: str) -> pd.DataFrame:
        """
        Initialize the extractor with the ID of the gene of interest.
        
        Parameters:
        gene_id (str): The ID of the gene to extract information for.
        """
        self.gene_id = gene_id
        self.mg = mygene.MyGeneInfo()
        self.go_info, self.entrez_id, self.ensembl_id, self.refseq_id, self.tax_id = self._get_go_and_ids_info()
        if self.go_info is not None:
            self.biological_process = self.go_info.get('BP', None)
            self.cellular_component = self.go_info.get('CC', None)
            self.molecular_function = self.go_info.get('MF', None)
            
        self.summary_table = self.get_dataframe()

    def _get_go_and_ids_info(self) -> list:
        """
        Retrieve GO information and IDs for the gene of interest.

        Returns:
        tuple: A tuple containing the GO information, Entrez ID, Ensembl ID, and RefSeq ID for the gene of interest.

        """
        try:
            gene_info = self.mg.query(self.gene_id, fields='go,entrezgene,ensembl.gene,refseq, taxid')
        except:
            return None, None, None, None, None
        if 'go' in gene_info:
            go_info = gene_info['go']
        elif 'hits' in gene_info:
            go_info = None
            for hit in gene_info['hits']:
                if 'go' in hit:
                    go_info = hit['go']
                    break
        

        entrez_id = gene_info.get('entrezgene')
        ensembl_id = gene_info.get('ensembl', {}).get('gene')
        refseq_id = gene_info.get('refseq')
        tax_id = gene_info.get('taxid')
        if not entrez_id and 'hits' in gene_info:
            for hit in gene_info['hits']:
                if 'entrezgene' in hit:
                    entrez_id = hit['entrezgene']
                    break

        if not ensembl_id and 'hits' in gene_info:
            for hit in gene_info['hits']:
                if 'ensembl' in hit and 'gene' in hit['ensembl']:
                    ensembl_id = hit['ensembl']['gene']
                    break

        if not refseq_id and 'hits' in gene_info:
            for hit in gene_info['hits']:
                if 'refseq' in hit:
                    refseq_id = hit['refseq']
                    break
        if not tax_id and 'hits' in gene_info:
            for hit in gene_info['hits']:
                if 'taxid' in hit:
                    tax_id = hit['taxid']
                    break

        if entrez_id is None:
            print(f"No Entrez ID found for gene ID {self.gene_id}")

        if ensembl_id is None:
            print(f"No Ensembl ID found for gene ID {self.gene_id}")

        if refseq_id is None:
            print(f"No RefSeq ID found for gene ID {self.gene_id}")

        if tax_id is None:
            print(f"No Taxonomy ID found for gene ID {self.gene_id}")
        return go_info, entrez_id, ensembl_id, refseq_id, tax_id



    def extract_terms(self, go_category: list) -> list:
        """
        Extract the terms from a GO category.
        
        Parameters:
        go_category (list): A list of dictionaries, each containing a GO category.
        
        Returns:
        list: A list of strings, each string being a term from the GO category.
        """
        return [item['qualifier'] + ' ' + item['term'] for item in go_category]

    def extract_go_ids(self, go_category: list) -> list:
        """
        Extract the GO IDs from a GO category.
        
        Parameters:
        go_category (list): A list of dictionaries, each containing a GO category.
        
        Returns:
        list: A list of strings, each string being a GO ID from the GO category.
        """
        return [item['id'] for item in go_category]
        
    def get_terms_and_go_ids(self):
        """
        Extract the terms and GO IDs from the biological process, cellular component,
        and molecular function categories of the gene of interest.
        
        Returns:
        tuple: A tuple containing six lists: the terms and GO IDs for the biological process,
        cellular component, and molecular function categories, respectively.
        
        Raises:
        ValueError: If no terms or GO IDs could be found for the gene of interest.
        """
        try:
            biological_process_terms = self.extract_terms(self.biological_process) if self.biological_process else []
            cellular_component_terms = self.extract_terms(self.cellular_component) if self.cellular_component else []
            molecular_function_terms = self.extract_terms(self.molecular_function) if self.molecular_function else []
            
            biological_process_go_id = self.extract_go_ids(self.biological_process) if self.biological_process else []
            cellular_component_go_id = self.extract_go_ids(self.cellular_component) if self.cellular_component else []
            molecular_function_go_id = self.extract_go_ids(self.molecular_function) if self.molecular_function else []
        except:
            print(f"No terms or GO IDs found for gene ID {self.gene_id}")
            biological_process_terms = 'None'
            cellular_component_terms = 'None'
            molecular_function_terms = 'None'
            biological_process_go_id = 'None'
            cellular_component_go_id = 'None'
            molecular_function_go_id = 'None'

        return biological_process_terms, cellular_component_terms, molecular_function_terms, \
            biological_process_go_id, cellular_component_go_id, molecular_function_go_id

    def get_dataframe(self):
        """
        Extract the terms, GO IDs, and gene identifiers from the biological process, cellular component,
        and molecular function categories of the gene of interest and put them into a DataFrame.
        
        Returns:
        DataFrame: A DataFrame containing the terms and GO IDs for the biological process,
        cellular component, and molecular function categories, as well as the gene identifiers.
        """
        biological_process_terms, cellular_component_terms, molecular_function_terms, \
        biological_process_go_id, cellular_component_go_id, molecular_function_go_id = self.get_terms_and_go_ids()

        df = pd.DataFrame({
            'biological_process_terms': pd.Series(biological_process_terms),
            'cellular_component_terms': pd.Series(cellular_component_terms),
            'molecular_function_terms': pd.Series(molecular_function_terms),
            'biological_process_go_id': pd.Series(biological_process_go_id),
            'cellular_component_go_id': pd.Series(cellular_component_go_id),
            'molecular_function_go_id': pd.Series(molecular_function_go_id),
            'tax_id': self.tax_id
        }).drop_duplicates()

        df['name'] = self.gene_id
        df['entrez_id'] = self.entrez_id
        df['ensembl_id'] = self.ensembl_id
        df['refseq_id'] = str(self.refseq_id)
        return df

import streamlit as st

class Gene(BaseModel):
    """A class for holding the gene information of each gene input"""
    name: str
    go_info: Optional[List[Dict[str, Any]]] = None

class GoFormater:
    """A class for coordinating the fetching of GO terms and their parsing."""

    def __init__(self, gene_list : list):
        """
        Initialize the GoFormater with the gene list that must be annotated.
        
        Parameters:
        gene_list (list): containing genes to be analysed
        """
        self.gene_list = gene_list
             
    def go_gene_annotation(self) -> pd.DataFrame:
        """
        Method that annotates genes and returns a data frame as followed:
        No Ensembl ID found for gene ID TUBB
       name                           biological_process_terms             cellular_component_terms                        molecular_function_terms biological_process_go_id cellular_component_go_id molecular_function_go_id  tax_id entrez_id       ensembl_id                                          refseq_id
        0    TUBB2B  involved_in microtubule cytoskeleton organization                   located_in nucleus                         enables GTPase activity               GO:0000226               GO:0005634               GO:0003924    9606    347733  ENSG00000137285  {'genomic': ['NC_000006.12', 'NC_060930.1', 'N...
        1    TUBB2B                     involved_in mitotic cell cycle               is_active_in cytoplasm  enables structural constituent of cytoskeleton               GO:0000278               GO:0005737               GO:0005200    9606    347733  ENSG00000137285  {'genomic': ['NC_000006.12', 'NC_060930.1', 'N...

        
        """
        def process_gene(gene: str):
            print(f"Processing gene: {gene}")
            dataframe = GeneGOExtractor(gene).get_dataframe()
            if dataframe is None:
                print(f"Could not process gene: {gene}")
                return None
            return Gene(
                name=dataframe.name[0], 
                go_info=dataframe.to_dict(orient="records")
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            genes_instances = list(executor.map(process_gene, self.gene_list))

        #Go processing and saving csv
        gene_dfs = [pd.DataFrame(gene.go_info) for gene in genes_instances]
        go_annotated_gene_df = pd.concat(gene_dfs, ignore_index=True)
        cols = ['name'] + [col for col in go_annotated_gene_df.columns if col != 'name']
        go_annotated_gene_df = go_annotated_gene_df[cols]

        return go_annotated_gene_df

    def identify_gene_column(self, df: pd.DataFrame):
        """Identify the column containing gene names based on a reference list."""
        
        def matches_in_column(column):
            """Return the number of matches in the given column."""
            return df[column].astype(str).str.contains('|'.join(self.gene_list)).sum()

        # First, always check the 'gene_name' column if it exists in the dataframe
        if 'gene_name' in df.columns:
            required_matches = max(1, min(5, max(len(self.gene_list)//50, df['gene_name'].shape[0])))
            if matches_in_column('gene_name') >= required_matches:
                return 'gene_name'
        
        for gene_matching_col in df.columns:
            # Skip the 'genemodel' column
            if gene_matching_col == 'genemodel' or gene_matching_col == 'gene_name':
                continue
            
            required_matches = max(1, min(5, max(len(self.gene_list)//50, df[gene_matching_col].shape[0])))
            # Check if the column contains the required number of reference gene names
            if matches_in_column(gene_matching_col) >= required_matches:
                return gene_matching_col
        return None

    def concatenate_dataframes(self,input_file_path: str):
        # Step 1: Read in the input_file_df from the provided path

        concat_gene_df = self.go_gene_annotation()
        input_file_df = pd.read_csv(input_file_path)

        # Step 2: Identify the gene column in input_file_df
        gene_col = self.identify_gene_column(input_file_df)
        if not gene_col:
            raise ValueError("No gene column identified in the input dataframe.")
        
        # Step 3: Split and explode the identified gene column directly
        input_file_df = input_file_df.assign(**{gene_col: input_file_df[gene_col].str.split('; ')}).explode(gene_col)
        
        # Step 4: Merge the two dataframes based on the gene names
        merged_df = pd.merge(input_file_df, 
                            concat_gene_df, 
                            left_on=gene_col, 
                            right_on='name', 
                            how='left').drop(columns='name')
        
        return merged_df


# @tool
# def go_file_tool(input_file_path: str, input_file_gene_name_column: str) -> pd.DataFrame:
#     """Used when the input is a file and not a human written query.
#     Tool to find gene ontologies (using mygene), outputs data frame with GO & gene id annotated gene names

#     Parameters:
#     input_file_path (str): A string which is a path to the csv file containing gene names to be annotated
#     input_file_gene_name_column (str): a string which is the column name containing the 

#     Returns:
#     final_dataframe (dataframe): A df containing annotated genes with GO & IDs from mygene concatenated with the input file.
#     """
    
#     gene_list = Utils.flatten_aniseed_gene_list(input_file_path, input_file_gene_name_column)

#     #we extract all the go terms and ids for all genes in this list
#     gof = GoFormater(gene_list)

#     final_go_df = gof.concatenate_dataframes(input_file_path)
#     final_go_df = Utils.parse_refseq_id(final_go_df)
#     final_go_df.to_csv(input_file_path[:-4]+'_GO_annotated.csv', index=False)
#     file_name = input_file_path[:-4]+'_GO_annotated.csv'
#     return final_go_df, file_name


@tool
def go_nl_query_tool(nl_input: str) -> pd.DataFrame:
    """Used when the input is a natural language written query containing gene names that need a GO annotation.
    Tool to find gene ontologies (using mygene), outputs data frame with GO & gene id annotated gene names

    Parameters:
    input_string (str): A natural language string containing gene names that have to be processed

    Returns:
    final_dataframe (dataframe): A df containing annotated genes with GO & IDs from mygene.
    """

    #we extract all the go terms and ids for all genes in this list

    extractor = NaturalLanguageExtractors(nl_input)
    gene_list = extractor.gene_name_extractor()
    gof = GoFormater(gene_list)
    final_go_df = gof.go_gene_annotation()
    final_go_df = Utils.parse_refseq_id(final_go_df)
    base_dir = os.getcwd() 
    SAVE_PATH = os.path.join(base_dir, 'baio', 'data', 'output', 'gene_ontology', 'go_annotation.csv')

    final_go_df.to_csv(SAVE_PATH)
    return final_go_df.head()

#stateless approach

def process_go_category(go_category):
    # Process a single GO category to extract terms and their IDs
    if isinstance(go_category, list) and all(isinstance(item, dict) for item in go_category):
        terms = [f"{item.get('qualifier', '')} {item.get('term', '')}".strip() for item in go_category]
        go_ids = [item.get('id', '') for item in go_category]
        return terms, go_ids
    else:
        return None, None

    
def unpack_go_terms(df):
    # Function to unpack GO terms and GO IDs
    df['biological_process_terms'], df['biological_process_go_id'] = zip(*df['go'].apply(lambda x: process_go_category(x.get('BP')) if isinstance(x, dict) else (None, None)))
    df['cellular_component_terms'], df['cellular_component_go_id'] = zip(*df['go'].apply(lambda x: process_go_category(x.get('CC')) if isinstance(x, dict) else (None, None)))
    df['molecular_function_terms'], df['molecular_function_go_id'] = zip(*df['go'].apply(lambda x: process_go_category(x.get('MF')) if isinstance(x, dict) else (None, None)))

    # Drop the original 'go' column
    df.drop('go', axis=1, inplace=True)
    return df


def unpack_refseq(df):
    # Unpack RefSeq data
    df['refseq_genomic'] = df['refseq'].apply(lambda x: x.get('genomic', None) if isinstance(x, dict) else None)
    df['refseq_protein'] = df['refseq'].apply(lambda x: x.get('protein', None) if isinstance(x, dict) else None)
    df['refseq_rna'] = df['refseq'].apply(lambda x: x.get('rna', None) if isinstance(x, dict) else None)

    # Drop the original 'refseq' column
    df.drop('refseq', axis=1, inplace=True)
    return df


def unpack_ensembl(df):
    # Unpack Ensembl data
    df['ensembl_gene'] = df['ensembl'].apply(lambda x: x.get('gene', '') if isinstance(x, dict) else '')

    # Drop the original 'ensembl' column
    df.drop('ensembl', axis=1, inplace=True)
    return df



def unpack_uniprot(df):
    # Unpack UniProt data
    df['uniprot_trEMBL'] = df['uniprot'].apply(lambda x: x.get('TrEMBL', None) if isinstance(x, dict) else None)
    df['uniprot_swissprot'] = df['uniprot'].apply(lambda x: x.get('Swiss-Prot', None) if isinstance(x, dict) else None)

    # Drop the original 'uniprot' column
    df.drop('uniprot', axis=1, inplace=True)
    return df


def get_gene_info_df(gene_name):
    mg = mygene.MyGeneInfo()
    response = mg.query(gene_name, fields='symbol,taxid,entrezgene,ensembl.gene,uniprot,refseq,go')

    # Directly create a DataFrame from the response
    df = pd.DataFrame(response['hits'])

    df = unpack_uniprot(df)
    df = unpack_go_terms(df)
    df = unpack_refseq(df)
    df = unpack_ensembl(df)
    # Add a new column with the gene name for all rows
    gene_query_col = [gene_name] * len(df)
    df.insert(0, 'query_name', gene_query_col)

    return df

def process_gene(gene):
    print(f"Processing gene: {gene}")
    try:
        gene_df = get_gene_info_df(gene)
        gene_df['gene_name'] = gene  # Adding a gene name column
        return gene_df
    except Exception as e:
        print(f"Could not process gene {gene}: {e}")
        return None
    
def process_genes(gene_list):
    gene_dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        for gene_df in executor.map(process_gene, gene_list):
            if gene_df is not None:
                gene_dfs.append(gene_df)

    # Concatenating all gene dataframes
    concatenated_df = pd.concat(gene_dfs, ignore_index=True)
    return concatenated_df


def identify_gene_column(df: pd.DataFrame, gene_list: list):
    """Identify the column containing gene names based on a reference list."""
    
    def matches_in_column(column):
        """Return the number of matches in the given column."""
        return df[column].astype(str).str.contains('|'.join(gene_list)).sum()

    # First, always check the 'gene_name' column if it exists in the dataframe
    if 'gene_name' in df.columns:
        required_matches = max(1, min(5, max(len(gene_list)//50, df['gene_name'].shape[0])))
        if matches_in_column('gene_name') >= required_matches:
            return 'gene_name'
    
    for gene_matching_col in df.columns:
        # Skip the 'genemodel' column
        if gene_matching_col == 'genemodel' or gene_matching_col == 'gene_name':
            continue
        
        required_matches = max(1, min(5, max(len(gene_list)//50, df[gene_matching_col].shape[0])))
        # Check if the column contains the required number of reference gene names
        if matches_in_column(gene_matching_col) >= required_matches:
            return gene_matching_col
    return None

def concatenate_dataframes(input_file_path: str, concat_gene_df, gene_list):
    # Step 1: Read in the input_file_df from the provided path

    input_file_df = pd.read_csv(input_file_path)

    # Step 2: Identify the gene column in input_file_df
    gene_col = identify_gene_column(input_file_df, gene_list)
    if not gene_col:
        raise ValueError("No gene column identified in the input dataframe.")
    
    # Step 3: Split and explode the identified gene column directly
    input_file_df = input_file_df.assign(**{gene_col: input_file_df[gene_col].str.split('; ')}).explode(gene_col)
    
    # Step 4: Merge the two dataframes based on the gene names
    merged_df = pd.merge(input_file_df, 
                        concat_gene_df, 
                        left_on=gene_col, 
                        right_on='query_name', 
                        how='left').drop(columns='query_name')
    
    return merged_df


def save_to_csv(df, filename):
    df.to_csv(filename, index=False)



def go_file_tool(input_file_path: str, input_file_gene_name_column: str) -> pd.DataFrame:
    """Used when the input is a file and not a human written query.
    Tool to find gene ontologies (using mygene), outputs data frame with GO & gene id annotated gene names

    Parameters:
    input_file_path (str): A string which is a path to the csv file containing gene names to be annotated
    input_file_gene_name_column (str): a string which is the column name containing the 

    Returns:
    final_dataframe (dataframe): A df containing annotated genes with GO & IDs from mygene concatenated with the input file.
    """
    
    gene_list = list(set(Utils.flatten_aniseed_gene_list(input_file_path, input_file_gene_name_column)))
    #we extract all the go terms and ids for all genes in this list
    final_go_df = process_genes(gene_list)
    final_go_df = concatenate_dataframes(input_file_path, final_go_df, gene_list)
    final_go_df.to_csv(input_file_path[:-4]+'_GO_annotated.csv', index=False)
    file_name = input_file_path[:-4]+'_GO_annotated.csv'
    return final_go_df, file_name


