from langchain.tools import BaseTool
from Bio import Entrez
from Bio import SeqIO
import os

from ..state import State
from sanclone.tools import settings


class ParseVirusTool(BaseTool):
    name = "parse_virus"
    description = "a tool that parses in the virus prompt"
    shared_state: State

    def _run(self, query: str) -> str:
        # Assume vector is vector name ParseVirusTool()._run('pET-16b') -> seq Record
        genbank_filename = get_vector_data(query, settings.OUTPUT_FOLDER)
        seqObj = list(SeqIO.parse(open(genbank_filename,"r"), "genbank"))[0]
        if seqObj is not None:
            self.shared_state.vector = seqObj
            return f"Vector {seqObj.description} is loaded. "
        else:
            return "Could not find Vector"


def get_vector_data(vector_name):
    # Set your email address for NCBI Entrez. This is required.
    output_folder = settings.OUTPUT_FOLDER
    Entrez.email = settings.email

    # Define the search query using the vector_name input
    search_query = vector_name

    # Use Entrez to search for GenBank records
    search_handle = Entrez.esearch(db="nucleotide", term=search_query)
    search_results = Entrez.read(search_handle)
    search_handle.close()

    # Check if any results were found
    if "IdList" not in search_results or not search_results["IdList"]:
        print(f"No GenBank records found for {vector_name}")
        return

    # Extract the first GenBank ID from the search results
    genbank_id = search_results["IdList"][0]

    # Download the GenBank record and save it to a file
    fetch_handle = Entrez.efetch(db="nucleotide", id=genbank_id, rettype="gb", retmode="text")
    genbank_record = SeqIO.read(fetch_handle, "genbank")
    fetch_handle.close()

    # Save the GenBank record to a file
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = os.path.join(output_folder, f"{genbank_record.id}.gbk")
    SeqIO.write(genbank_record, filename, "genbank")

    #print(f"Downloaded GenBank file for {vector_name} to {filename}")
    return filename


def get_genbank_from_soup(query):
    return None