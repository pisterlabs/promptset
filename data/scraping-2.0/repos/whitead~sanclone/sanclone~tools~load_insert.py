from Bio import Entrez, SeqIO
import json


from langchain.tools import BaseTool

from ..state import State


class ParseGeneTool(BaseTool):
    name = "parse_genes"
    description = "a tool that parses in the virus prompt"
    shared_state: State

    def _run(self, query: str) -> str:
        # Assume query is a json object of the form {"gene_name": "gene_name", "organism": "organism"}
        qson = json.loads(query)
        gene_name = qson['gene_name']
        organism = qson['organism']
        seq_record = fetch_sequence(gene_name, organism)
        if seq_record is not None:
            self.shared_state.linear_insert = seq_record
            return f"Sequence {seq_record.description} is loaded. "
        else:
            return "Could not find Sequence"
    
def fetch_sequence(gene_name, organism):
    Entrez.email = "your.email@example.com"  # Always tell NCBI who you are
    search_term = f"{gene_name}[Gene Name] AND {organism}[Organism] AND mRNA[Filter]"

    # Search for the gene's mRNA ID
    handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=1)
    record = Entrez.read(handle)
    handle.close()

    if not record["IdList"]:
        # print("No sequence found!")
        return None

    gene_id = record["IdList"][0]

    # Fetch the sequence based on the ID
    handle = Entrez.efetch(db="nucleotide", id=gene_id, rettype="fasta", retmode="text")
    seq_record = SeqIO.read(handle, "fasta")
    handle.close()

    return seq_record