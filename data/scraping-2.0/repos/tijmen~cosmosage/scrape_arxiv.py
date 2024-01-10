#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods for interacting with papers from the ArXiV.

There are several methods for manually detexifying and cleaning up the papers.

An alternative method relies on langchain and VLLM to generate summaries and 
QA pairs from the ArXiV papers. This interacts with the paper PDFs
rather than the .tex files. See the arxiv_paper class for more details.

Author: Tijmen de Haan <tijmen.dehaan@gmail.com>
"""
import io
import tarfile
import os
import requests
from bs4 import BeautifulSoup
import re
import requests
import urllib.request
import json
import urllib.request
import os
import xml.etree.ElementTree as ET
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field


def get_arxiv_ids(search_params):
    arxiv_ids_all = []
    max_iterations = 100  # don't look for more papers than max_iterations*max_results
    max_results = 1000  # per iteration
    for i in range(max_iterations):
        search_params["max_results"] = max_results
        search_params["start"] = 1000 * i
        search_params["sortBy"] = "submittedDate"
        search_params["sortOrder"] = "descending"
        # arXiv API endpoint
        url = "http://export.arxiv.org/api/query"

        # Request to arXiv API
        print(f"requesting from URL: {url} with params: {search_params}")
        response = requests.get(url, params=search_params)

        # Check if the request was successful
        if response.status_code != 200:
            raise Exception("Failed to retrieve data from arXiv")

        # Parse the response using BeautifulSoup with xml as the XML parser
        soup = BeautifulSoup(response.content, features="xml")

        # Extract arXiv IDs from the response
        arxiv_ids = [entry.id.text.split("/")[-1] for entry in soup.find_all("entry")]

        # remove the version numbers
        arxiv_ids = [re.sub("v[0-9]+", "", arxiv_id) for arxiv_id in arxiv_ids]

        if arxiv_ids:
            arxiv_ids_all.extend(arxiv_ids)
        else:
            return arxiv_ids_all


def extract_tex(papers, tex_files_path):
    """
    Download papers from the arxiv. It downloads the tar file, extracts the .tex
    file, and saves it to disk.

    Args:
        papers (list): A list of paper IDs.
        tex_files_path (str): The path to the folder where the .tex files will be saved.

    Returns:
        None
    """
    for ipaper, paper in enumerate(papers):
        output_file_path = os.path.join(tex_files_path, paper + ".tex")
        if os.path.exists(output_file_path):
            continue

        print(f"Fetching paper {ipaper} of {len(papers)}.")

        # fetch paper from arxiv
        url = "https://arxiv.org/e-print/" + paper
        # resp = get_response(search_query=paper, start_index=1)

        # send GET request to the URL of the paper
        response = requests.get(url, timeout=10)
        tar_bytes = response.content

        # extract tarball, retain .tex file and delete the rest
        try:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
                tex_file_name = None
                # iterate over tar.getmembers() to find the one that ends in .tex
                for member in tar.getmembers():
                    if member.name.endswith(".tex"):
                        tex_file_name = member.name
                        break

                if tex_file_name:
                    tex_file = tar.extractfile(tex_file_name)
                    try:
                        tex_content = tex_file.read().decode("utf-8")
                    except Exception as e:
                        print(f"Error reading tex file for paper {paper}: {e}")
                        tex_content = ""
                else:
                    print(f"No .tex file found in the tar archive for paper {paper}")
                    tex_content = ""

            del tar_bytes
        except Exception as e:
            print(f"Error extracting tar archive for paper {paper}: {e}")
            # create a blank .tex file
            tex_content = ""

        # now save the tex file to disk
        if tex_file_name:
            with open(
                os.path.join(tex_files_path, paper + ".tex"), "w", encoding="utf-8"
            ) as file:
                file.write(tex_content)


def other_arxiv_recommendation_ids():
    # more arxiv papers recommended for me by asl (but no tags)
    return [
        "2011.08163",
        "2106.11202",
        "1907.11976",
        "1809.00036",
        "1907.10947",
        "1407.2973",
        "1809.00033",
        "1902.09640",
        "1911.08047",
        "2103.16017",
        "1904.12995",
        "1809.00032",
        "1805.03219",
        "1803.10682",
        "1910.05748",
        "2008.12619",
        "2111.14785",
        "2012.01709",
        "2101.01684",
        "2002.06197",
        "1905.05777",
        "2103.06166",
        "1704.00884",
        "2308.11608",
        "1807.02199",
        "1908.01062",
        "1907.04473",
        "1412.4760",
        "1808.00568",
        "1408.3161",
        "1907.08605",
        "1411.1042",
        "2006.08061",
        "2110.00482",
        "1705.00743",
        "1412.7521",
        "2103.13618",
        "1503.02315",
        "2208.08559",
        "2207.11937",
        "1604.03507",
        "1708.01360",
        "1907.09621",
        "2203.08024",
        "1909.01305",
        "1810.10643",
        "2003.03431",
        "1712.07541",
        "1605.00966",
        "2111.07491",
        "2310.10849",
        "1307.2903",
        "1801.06987",
        "2012.04047",
        "2002.05254",
        "1809.00030",
        "1701.04396",
        "1810.02342",
        "2002.05228",
        "2106.14797",
        "2203.16556",
        "2102.00809",
        "1706.10286",
        "2101.06342",
        "2002.05219",
        "2012.05934",
        "2002.05197",
        "2111.04631",
        "2102.05033",
        "2012.04532",
        "2111.04816",
        "1907.02156",
        "1407.3161",
        "1808.07445",
        "1810.02441",
        "1512.07663",
        "2208.02284",
        "2111.14751",
        "1910.07157",
        "2203.12439",
        "1607.04668",
        "1603.06522",
        "2207.14796",
        "1808.00567",
        "2101.12449",
        "1810.02212",
        "2111.04633",
        "2209.09864",
        "1907.08284",
        "1311.4953",
        "2206.10824",
        "1706.02464",
        "1910.04121",
        "2208.01080",
        "1409.0850",
        "2203.12440",
        "1707.09353",
        "1810.10998",
        "2202.02773",
        "2001.01724",
        "1212.6267",
        "2111.04778",
        "2112.02425",
        "2309.09908",
        "2212.05642",
        "1512.07299",
        "2102.02386",
        "1601.00125",
        "1912.04272",
        "2203.07638",
        "1908.07642",
        "2112.03606",
        "2311.01846",
        "2203.16567",
        "1610.02743",
        "1303.3535",
        "2210.08038",
        "2210.08633",
        "1410.7488",
        "1608.03025",
        "2210.05684",
        "2012.07077",
        "2210.04117",
        "1808.00569",
        "2304.01158",
        "2012.08547",
        "2302.05228",
        "1812.01679",
        "1810.02322",
        "2101.05306",
        "1603.03904",
        "1602.07384",
        "2304.05203",
        "2207.11804",
        "1808.00570",
        "2111.11495",
        "1601.05452",
        "2110.00483",
        "1607.06064",
        "2304.05202",
        "2204.05869",
        "2311.04424",
        "2304.00973",
        "2310.07657",
        "2202.01324",
        "2207.11377",
        "2307.01258",
        "1609.05211",
        "2208.10482",
        "1805.09346",
        "2209.12492",
        "2210.10893",
        "1912.00860",
        "2302.04297",
        "2012.09363",
        "2206.03389",
        "2111.07742",
        "1809.03689",
        "1407.2942",
        "1607.04567",
        "2211.03786",
        "1407.6894",
        "1808.00571",
        "2202.10055",
        "1607.06861",
        "2311.07512",
        "2211.00542",
        "2204.12503",
        "1407.7520",
        "2301.09634",
        "2012.07862",
        "1805.00470",
        "2210.16202",
        "1502.00619",
        "1803.01018",
        "2003.08949",
        "1807.05995",
        "1912.12782",
        "1403.4302",
        "2201.04507",
        "1711.02594",
        "2007.07289",
        "1907.09035",
        "2011.03483",
        "1404.6250",
        "1710.08456",
        "1801.06991",
        "1711.04169",
        "1407.5928",
        "2306.05460",
        "1502.00596",
        "2108.03316",
        "2101.02658",
        "2207.11374",
        "1808.10491",
        "2208.02755",
        "1808.04493",
        "2111.11301",
        "1904.01640",
        "1806.05576",
        "2112.01458",
        "2007.07288",
        "1309.5381",
        "1807.07496",
        "1808.10037",
        "1309.5383",
        "2010.07998",
        "2203.07246",
        "2208.05997",
        "1909.11569",
        "1512.04535",
        "2311.05793",
        "1810.05216",
        "1606.01968",
        "2207.11375",
        "2207.13737",
        "1403.3985",
        "1507.05551",
        "1501.07911",
        "2208.10523",
        "1910.10199",
        "1606.09396",
        "2208.14159",
        "1506.07814",
        "1802.03822",
        "2001.07848",
        "2310.12213",
        "1510.02809",
        "1705.02907",
        "1805.08363",
        "2101.09608",
        "2101.11917",
        "1605.08770",
        "1705.02523",
        "1502.00643",
        "1610.02360",
        "1601.07923",
        "1702.07020",
        "1802.05257",
        "1510.09217",
        "1807.01384",
        "1401.8029",
        "1707.02981",
        "2111.09140",
        "2207.10012",
        "2311.05583",
        "1911.11902",
        "2007.07290",
        "1809.06556",
        "1710.04326",
        "1912.02902",
        "1809.07373",
        "2001.02763",
        "2104.09511",
        "1911.05717",
        "2005.06168",
        "2101.08374",
        "2207.13204",
        "1910.02608",
        "2009.07772",
        "1908.00480",
        "2112.00820",
        "2012.10345",
        "1807.05215",
        "1407.2584",
        "1711.02266",
        "1312.6645",
        "1312.6646",
        "1806.04316",
        "2009.05557",
        "2304.05196",
        "1602.07744",
        "2304.09166",
        "2103.02747",
        "1603.05976",
        "2103.03154",
        "1711.02523",
        "1608.08891",
        "1405.5524",
        "1607.05754",
        "2307.12931",
        "1605.06569",
        "1711.10596",
        "2207.14242",
        "1605.05329",
        "2102.02129",
        "1608.08234",
        "2112.07656",
        "1705.00411",
        "2207.14212",
        "1607.01825",
        "1801.02543",
        "1711.04841",
        "1403.2369",
        "1509.02461",
        "2009.08822",
        "2101.08373",
        "2101.10298",
        "2101.03833",
        "2303.12345",
        "1611.09753",
        "1711.05344",
        "2004.11601",
        "2306.08875",
        "2012.08636",
        "1908.08057",
        "1710.11255",
        "1808.05152",
        "2111.01797",
        "1808.01592",
        "1408.4790",
        "2209.02708",
        "2203.02495",
        "1911.10980",
        "1707.01488",
        "1710.02239",
        "2012.12407",
        "1808.01349",
        "1412.0626",
        "1604.02593",
        "2002.05771",
        "1509.06770",
        "2310.00059",
        "2101.08410",
        "2208.02854",
        "1310.1422",
        "2209.12672",
        "1807.00058",
        "1807.03927",
        "2001.10465",
        "2311.00315",
        "2205.04494",
        "2102.06092",
        "1811.06081",
        "2102.03661",
        "2208.12604",
        "1608.06262",
        "1904.02116",
        "1709.05600",
        "1402.3601",
        "2006.06594",
        "2010.13800",
        "1903.07046",
        "2111.01055",
        "2004.01139",
        "1511.05036",
        "2107.00473",
        "2009.07591",
        "2108.01663",
        "1702.01871",
        "1311.5388",
        "1910.07456",
        "2103.13334",
        "2007.04325",
        "1909.13832",
        "2205.06901",
        "2011.02449",
        "1607.03796",
        "1611.03866",
        "2212.01370",
        "2204.01885",
        "1903.04763",
        "1509.04714",
        "2106.12467",
        "2103.05582",
    ]


class arxiv_paper:
    """
    Class to hold an arxiv paper and its metadata.
    Can generate an overall summary and QA pairs with an LLM.
    """

    def __init__(self, arxiv_id):
        self.arxiv_id = arxiv_id

        # connect to the VLLM server that I started separately with something like
        #  python -u -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --model mistralai/Mistral-7B-Instruct-v0.2

        inference_server_url = "http://0.0.0.0:8000/v1"

        self.llm = ChatOpenAI(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            openai_api_key="EMPTY",
            openai_api_base=inference_server_url,
            max_tokens=4096,
            temperature=0.4,
        )
        self.fetch_paper_data()
        self.title = self.paper_data.find("{http://www.w3.org/2005/Atom}title").text
        self.year = self.paper_data.find(
            "{http://www.w3.org/2005/Atom}published"
        ).text.split("-")[0]
        self.first_author = (
            self.paper_data.find("{http://www.w3.org/2005/Atom}author")
            .find("{http://www.w3.org/2005/Atom}name")
            .text
        )
        self.shorthand_title = f"{self.first_author} et al. ({self.year})"
        self.pdf_filepath = f"datasets/arxiv_pdf/{self.arxiv_id}.pdf"
        self.download_pdf()
        self.load_pdf()

    def fetch_paper_data(self):
        url = f"http://export.arxiv.org/api/query?id_list={self.arxiv_id}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("Error fetching paper data")

        try:
            root = ET.fromstring(response.content)
            self.paper_data = root.find("{http://www.w3.org/2005/Atom}entry")
        except ET.ParseError:
            raise Exception("Error parsing XML data")

    def download_pdf(self):
        if not os.path.exists(self.pdf_filepath):
            pdf_url = f"http://arxiv.org/pdf/{self.arxiv_id}.pdf"
            urllib.request.urlretrieve(pdf_url, self.pdf_filepath)

    def load_pdf(self):
        loader = PyPDFLoader(self.pdf_filepath)
        self.pages = loader.load()

        def clean_up(doc):
            # weird double f character
            doc = doc.replace(chr(64256), "ff")
            # -\n are hyphenated line breaks which need to be removed altogether
            doc = doc.replace("-\n", "")
            # double line breaks are probably paragraph breaks
            doc = doc.replace("\n\n", "<|paragraph_break|>")
            # remaining line breaks can be replaced by spaces
            doc = doc.replace("\n", " ")
            # put back double line breaks
            doc = doc.replace("<|paragraph_break|>", "\n\n")
            return doc

        # clean up pages
        for page in self.pages:
            page.page_content = clean_up(page.page_content)

    def generate_summary(self):
        summarize_chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        self.summary = summarize_chain.run(self.pages)
        return self.summary

    def generate_qa_pairs(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        texts = text_splitter.split_documents(self.pages)

        class question_answer(BaseModel):
            question: str = Field(..., description="Question framed.")
            answer: str = Field(..., description="Answer to the question.")

        class output(BaseModel):
            output: list[question_answer] = []

        parser = PydanticOutputParser(pydantic_object=output)

        # make sure summary has been generated
        if not hasattr(self, "summary"):
            self.generate_summary()

        prompt = (
            """You are an expert on cosmology and are tasked with generating questions and answers. You make question-answer pairs from a given PASSAGE of a scientific paper. The question should be understood BY ITSELF, so include the shorthand form, title, or arXiv ID if referring to a specific context. The answer should be long and demonstrate an excellent understanding of the subject matter.
        arXiv ID: """
            + self.arxiv_id
            + """
 
        Paper title: """
            + self.title
            + """

        Shorthand title: """
            + self.shorthand_title
            + """    
            
        Overall paper summary: """
            + self.summary
            + """

        PASSAGE: {text}

        {format_instructions}

        Output:"""
        )

        self.qa_pairs = []

        for text in texts:
            _prompt = PromptTemplate(
                template=prompt,
                input_variables=["text"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            _input = _prompt.format_prompt(text=text)
            message = [HumanMessage(content=_input.to_string())]

            result = self.llm(message).content
            try:
                json_result = json.loads(result)
            except json.decoder.JSONDecodeError as e:
                print("Cannot serialize this output because of JSONDecodeError", e)
                continue
            for json_result in json_result:
                if "question" in json_result and "answer" in json_result:
                    self.qa_pairs.append(json_result)
                elif "output" in json_result:
                    for item in json_result["output"]:
                        self.qa_pairs.append(item)
                else:
                    continue

        return self.qa_pairs

    def save_dataset_jsonl(self):
        with open(f"datasets/arxiv_qa/{self.arxiv_id}.jsonl", "w") as f:
            # first save the summary
            summary = {
                "question": f"Summarize {self.shorthand_title}.",
                "answer": f'{self.shorthand_title} is titled "{self.title}" and has ArXiV ID {self.arxiv_id}. {self.summary}',
            }
            f.write(json.dumps(summary) + "\n")

            # then save the qa_pairs
            for item in self.qa_pairs:
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    paper = arxiv_paper("1603.06522")
    paper.generate_summary()
    paper.generate_qa_pairs()
    paper.save_dataset_jsonl()
