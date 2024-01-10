import json
import os
import xml.etree.ElementTree as ET

import openai

import AutoGPT.AutoGPT as AutoGPT

openai.organization = "org-bWPRxgvD4e3jFTVqMKVBiGMp"
openai.api_key = os.getenv("OPENAI_API_KEY")


class MYGENE(AutoGPT.Tool):
    mygene_api = """API PARAMETERS - query: What is being searched"""

    mygene_example = """
import mygene
mg = mygene.MyGeneInfo()

query_term="{}"
fields="symbol,name,entrezgene,ensemblgene"
size=10
from_=0

gene_results = mg.query(query_term, fields=fields, species="human", size=size, from_=from_)
hits = gene_results['hits']

gene_info_list = []

for gene in hits:
    gene_info = mg.getgene(gene['_id'], fields=['name','symbol','type_of_gene','genomic_pos_hg19','refseq','taxid', 
    'generif','summary','pathway'])
    gene_info_list.append(gene_info)

ret = gene_info_list
"""

    def __init__(self):
        schema = {"query": {"title": "query", "type": "string"}}
        super().__init__("MYGENE", "This is useful for finding information on specific genes, or genes associated "
                                   "with the search query. If you wish to make a task to create an API request to mygene then simply say 'MYGENE:' followed by what you would like to search for. Example: 'MYGENE: look up information on genes that are linked to cancer'" + self.mygene_api,
                         schema, self.execute_gen_python)

    def execute_gen_python(self, args):
        results = {}
        exec(self.mygene_example.format(args["query"]), globals(), results)
        processed_result = []
        for json_data in results["ret"]:

            name = json_data.get("name")
            refseq_genomic = json_data.get("refseq", {}).get("genomic", [])
            refseq_rna = json_data.get("refseq", {}).get("rna", [])
            symbol = json_data.get("symbol")
            taxid = json_data.get("taxid")
            type_of_gene = json_data.get("type_of_gene")
            pos = json_data.get("genomic_pos_hg19")
            summary = json_data.get("summary")
            generif = json_data.get("generif")

            output_summary = ""
            citation_data = ""

            # Summary
            if name:
                output_summary += f"Gene Name: {name}\n"
            if refseq_genomic:
                output_summary += f"RefSeq genomic: {', '.join(refseq_genomic)}\n"
            if refseq_rna:
                output_summary += f"RefSeq rna: {', '.join(refseq_rna)}\n"
            if symbol:
                output_summary += f"Symbol: {symbol}\n"
            if taxid:
                output_summary += f"Tax ID: {taxid}\n"
            if type_of_gene and type_of_gene != 'unknown':
                output_summary += f"Type of gene: {type_of_gene}\n"
            if pos:
                output_summary += f"Position: {pos}\n"
            if summary:
                output_summary += f"Summary of {name}: {summary}\n"
            else:
                if generif:
                    for rif in generif[:10]:
                        pubmed = rif.get("pubmed")
                        text = rif.get("text")

                        if text:
                            output_summary += text

                            if pubmed:
                                citation_data += f" Pubmed ID: {pubmed}"

            output_summary = output_summary.strip()

            if output_summary:
                processed_result.append((output_summary, {"citation_data": citation_data}))

        return processed_result


class MYVARIANT(AutoGPT.Tool):
    myvariant_api = """API PARAMETERS - query: What is being searched"""

    myvariant_example = """
import myvariant
mv = myvariant.MyVariantInfo()
query_term = '{}'
ret = mv.getvariant(query_term)"""

    def __init__(self):
        schema = {"query": {"title": "query", "type": "string"}}
        super().__init__("MYVARIANT",
                         """This is useful for finding information on specific genetic variants. If you wish to make a task to create an API request to myvariant then simply say 'MYVARIANT:' followed by the specific genetic variant you are interested in. You can specify by rsID, ClinVar, or in a standardized format for describing a genetic variant.Example: 'MYVARIANT: chr7:g.140453134T>C"""
                         + self.myvariant_api, schema,
                         self.execute_gen_python)

    def execute_gen_python(self, args):
        results = {}
        exec(self.myvariant_example.format(args["query"]), globals(), results)
        processed_result = []

        if not isinstance(results, list):
            results = [results]

        for result in results:
            variant_name = result.get("_id")
            gene_affected = result.get("cadd", {}).get("gene", {}).get("genename")
            consequence = result.get("cadd", {}).get("consequence")
            cadd_score = result.get("cadd", {}).get("phred")
            rsid = result.get("dbsnp", {}).get("rsid")

            variant_data = ""
            citation_data = ""

            if variant_name:
                variant_data += f"Variant Name: {variant_name}\n"
            if gene_affected:
                variant_data += f"Gene Affected: {gene_affected}\n"
            if consequence:
                variant_data += f"Consequence: {consequence}\n"
            if cadd_score is not None:
                variant_data += f"CADD Score: {cadd_score}\n"
            if rsid:
                variant_data += f"rsID: {rsid}\n"

            processed_result.append((variant_data, {"citation_data": citation_data}))

        return processed_result


class PUBMED(AutoGPT.Tool):
    pumbed_api = """API PARAMETERS - query: What is being searched"""
    pumbed_example = """
from Bio import Entrez
Entrez.email = 'saatvikramani777@gmail.com'

query_term = "{}"
retmax=6
retstart=0

search_handle = Entrez.esearch(db="pubmed", term=query_term, retmax=retmax, retstart=retstart)
search_results = Entrez.read(search_handle)
search_handle.close()

pubmed_ids = search_results["IdList"]

fetch_handle = Entrez.efetch(db="pubmed", id=pubmed_ids, rettype="abstract")
abstracts = fetch_handle.read()
fetch_handle.close()

ret = abstracts
"""

    def __init__(self):
        schema = {"query": {"title": "query", "type": "string"}}
        super().__init__("PUBMED",
                         "This is useful for searching biomedical literature and studies on any medical subject."
                         + self.pumbed_api, schema,
                         self.execute_gen_python)

    def execute_gen_python(self, args):
        result = {}
        exec(self.pumbed_example.format(args["query"]), globals(), result)
        try:
            root = ET.fromstring(result["ret"])
        except Exception as e:
            print(f"Cannot parse pubmed result, expected xml. {e}")
            print("Adding whole document. Note this will lead to suboptimal results.")
            return result if isinstance(result, list) else [result]

        processed_result = []

        for article in root:
            res_ = ""
            citation_data = ""
            for title in article.iter("Title"):
                res_ += f"{title.text}\n"
                citation_data += f"{title.text}\n"
            for abstract in article.iter("AbstractText"):
                res_ += f"{abstract.text}\n"
            for author in article.iter("Author"):
                try:
                    citation_data += f"{author.find('LastName').text}"
                    citation_data += f", {author.find('ForeName').text}\n"
                except:
                    pass
            for journal in article.iter("Journal"):
                res_ += f"{journal.find('Title').text}\n"
                citation_data += f"{journal.find('Title').text}\n"
            for volume in article.iter("Volume"):
                citation_data += f"{volume.text}\n"
            for issue in article.iter("Issue"):
                citation_data += f"{issue.text}\n"
            for pubdate in article.iter("PubDate"):
                try:
                    year = pubdate.find("Year").text
                    citation_data += f"{year}"
                    month = pubdate.find("Month").text
                    citation_data += f"-{month}"
                    day = pubdate.find("Day").text
                    citation_data += f"-{day}\n"
                except:
                    pass
            for doi in article.iter("ELocationID"):
                if doi.get("EIdType") == "doi":
                    res_ += f"{doi.text}\n"

            if res_:
                processed_result.append((res_, {"citation_data": citation_data}))

        return processed_result


def summarize(text):
    res = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                       messages=[{"role": "user", "content": f"{text}\n\nTl;dr"}], temperature=0.0)
    return res.choices[0].message.content


if __name__ == "__main__":
    m = AutoGPT.Memory()
    tools = [MYGENE(), PUBMED(), MYVARIANT(), AutoGPT.WriteFile(), AutoGPT.ReadFile()]
    agent = AutoGPT.Agent(
        tools,
        ["Write brief description on Breast cancer"],
        m
    )
    agent.run()
