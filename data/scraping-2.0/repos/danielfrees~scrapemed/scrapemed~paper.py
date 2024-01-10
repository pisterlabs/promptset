"""
ScrapeMed's Paper Module
============================

The scrapemed `paper` module is intended as the primary point of contact for
scrapemed end users.

Paper objects are defined here, as well end-user functionality for scraping
data from PubMed Central without stressing about the details.

..warnings::
    - :class:`emptyTextWarning` - Warned when trying to perform a text
        operation on a Paper which has no text.
    - :class:`pubmedHTTPError` - Warned when unable to retrieve a PMC XML
        repeatedly. Can occasionally happen with PMC due to high traffic.
        Also may be caused by broken XML formatting.
"""

import scrapemed._parse as parse
import lxml.etree as ET
import pandas as pd
import datetime
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from typing import Union, Dict
from difflib import SequenceMatcher
import uuid
import re
import warnings
from urllib.error import HTTPError
import time


class emptyTextWarning(Warning):
    """
    Warned when trying to perform a text operation on a Paper which has no text.
    """

    pass


class pubmedHTTPError(Warning):
    """
    Warned when unable to retrieve a PMC XML repeatedly. Can occasionally
    happen with PMC due to high traffic. Also may be caused by broken XML
    formatting.
    """

    pass


# --------------------PAPER OBJECT SCHEMA-------------------------------------
class Paper:
    """
    Class for storing paper data downloaded from PMC.

    This class provides methods for initializing papers via PMCID and directly
    from XML, paper chunking and vectorization, conversion to relational format
    (pandas Series), printing methods, and equality checking.

    Class data members include all of the data defined via the method
    :meth:`~Paper.info`.

    :raises pubmedHTTPError: Raised if there are HTTP errors when
        retrieving data from PMC.
    :raises emptyTextWarning: Raised if an attempt is made to vectorize a paper
        with no text.

    :Example:

    To initialize a Paper object with paper_dict:

    >>> paper = Paper(paper_dict)
    """

    __tablename__ = "Papers"

    def __init__(self, paper_dict: dict) -> None:
        """
        Initialize a Paper object with paper information parsed from a PMC download.

        :param dict paper_dict: A dictionary containing paper information, typically
            obtained from the parse.generate_paper_dict method.
        """
        if not paper_dict:
            self.has_data = False
            return None
        else:
            self.has_data = True

        # capture current time as time of last update. Note that this date
        # may not be synced with PMC paper updates if using
        # initialization via Paper.from_xml. Use Paper.from_pmc to update
        # papers directly via PMC
        current_datetime = datetime.datetime.now()
        current_year = current_datetime.year
        current_month = current_datetime.month
        current_day = current_datetime.day
        self.last_updated = (current_month, current_day, current_year)

        # read in the Paper data from the parsed paper_dict
        self.pmcid = paper_dict["PMCID"]
        self.title = paper_dict["Title"]
        self.authors = paper_dict["Authors"]
        self.non_author_contributors = paper_dict["Non-Author Contributors"]
        self.abstract = paper_dict["Abstract"]
        self.body = paper_dict["Body"]
        self.journal_id = paper_dict["Journal ID"]
        self.journal_title = paper_dict["Journal Title"]
        self.issn = paper_dict["ISSN"]
        self.publisher_name = paper_dict["Publisher Name"]
        self.publisher_location = paper_dict["Publisher Location"]
        self.article_id = paper_dict["Article ID"]
        self.article_types = paper_dict["Article Types"]
        self.article_categories = paper_dict["Article Categories"]
        self.published_date = paper_dict["Published Date"]
        self.volume = paper_dict["Volume"]
        self.issue = paper_dict["Issue"]
        self.fpage = paper_dict["First Page"]
        self.lpage = paper_dict["Last Page"]
        self.permissions = paper_dict["Permissions"]
        if self.permissions:
            self.copyright = self.permissions["Copyright Statement"]
            self.license = self.permissions["License Type"]
        else:
            self.copyright = None
            self.license = None
        self.funding = paper_dict["Funding"]
        self.footnote = paper_dict["Footnote"]
        self.acknowledgements = paper_dict["Acknowledgements"]
        self.notes = paper_dict["Notes"]
        self.custom_meta = paper_dict["Custom Meta"]
        self.ref_map = paper_dict["Ref Map"]
        self._ref_map_with_tags = paper_dict["Ref Map With Tags"]
        self.citations = paper_dict["Citations"]
        self.tables = paper_dict["Tables"]
        self.figures = paper_dict["Figures"]

        self.data_dict = parse.define_data_dict()

        self.vector_collection = None

        return None

    @classmethod
    def from_pmc(
        cls,
        pmcid: int,
        email: str,
        download: bool = False,
        validate: bool = True,
        verbose: bool = False,
        suppress_warnings: bool = False,
        suppress_errors: bool = False,
    ):
        """
        Generate a Paper from a PMCID with optional parameters.

        :param int pmcid: Unique PMCID for the article to parse.
        :param str email: Provide your email address for authentication with PMC.
        :param bool download: Whether or not to download the XML retrieved from PMC.
        :param bool validate: Whether or not to validate the XML from PMC against NLM
            articleset 2.0 DTD (HIGHLY RECOMMENDED).
        :param bool verbose: Whether or not to have verbose output for testing.
        :param bool suppress_warnings: Whether to suppress warnings while parsing XML.
            Note: Warnings are frequent, because of the variable nature of PMC
            XML data. Recommended to suppress when parsing many XMLs at once.
        :param bool suppress_errors: Return None on failed XML parsing, instead of
            raising an error.

        :return: A Paper object initialized via the passed PMCID and
            optional parameters.
        :rtype: Paper
        """
        NUM_TRIES = 3
        paper_dict = None
        for i in range(NUM_TRIES):
            try:
                paper_dict = parse.paper_dict_from_pmc(
                    pmcid=pmcid,
                    email=email,
                    download=download,
                    validate=validate,
                    verbose=verbose,
                    suppress_warnings=suppress_warnings,
                    suppress_errors=suppress_errors,
                )
                break
            except HTTPError:
                time.sleep(5)
        if not paper_dict:
            warnings.warn(
                (
                    f"Unable to retrieve PMCID {pmcid} from PMC. May be due to "
                    "HTTP traffic or broken XML formatting, try again later if "
                    "the former."
                ),
                pubmedHTTPError,
            )
            return None
        return cls(paper_dict)

    @classmethod
    def from_xml(
        cls,
        pmcid: int,
        root: ET.Element,
        verbose: bool = False,
        suppress_warnings: bool = False,
        suppress_errors: bool = False,
    ):
        """
        Generate a Paper straight from PMC XML.

        :param int pmcid: PMCID for the XML. THis is required intentionally,
            to ensure trustworthy unique indexing of PMC XMLs.
        :param ET.Element root: Root element of the PMC XML tree.
        :param bool verbose: Report verbose output or not. Intended for testing.
        :param bool suppress_warnings: Suppress warnings while
            parsing XML or not.
            Note: Warnings are frequent, because of the variable nature of
            PMC XML data.
            Recommended to suppress when parsing many XMLs at once.
        :param bool suppress_errors: Return None on failed XML parsing,
            instead of raising an error.
            Recommended to suppress when parsing many XMLs at once, unless
            failure is not an option.

        :returns: A Paper object initialized via the passed XML.
        :rtype: Paper
        """
        paper_dict = parse.generate_paper_dict(
            pmcid,
            root,
            verbose=verbose,
            suppress_warnings=suppress_warnings,
            suppress_errors=suppress_errors,
        )
        return cls(paper_dict)

    def info(self) -> Dict[str, str]:
        """
        Return the data definition dictionary.

        :return: A dictionary containing paper information.
        :rtype: dict[str, str]
        """
        return self.data_dict

    def print_abstract(self) -> str:
        """
        Print and return a string representation of the abstract.

        :return: A string containing the abstract text.
        :rtype: str
        """
        s = self.abstract_as_str()
        print(s)
        return s

    def abstract_as_str(self) -> str:
        """
        Return a string representation of the abstract of a paper.

        This method retrieves the abstract text without MHTML data references.

        :return: A string containing the abstract text.
        :rtype: str
        """
        s = ""
        if self.abstract:
            for sec in self.abstract:
                s += "\n"
                s += str(sec)
        return s

    def print_body(self) -> str:
        """
        Print and return a string representation of the body of a paper.

        This method retrieves the body text without MHTML data references.

        :return: A string containing the body text.
        :rtype: str
        """
        s = self.body_as_str()
        print(s)
        return s

    def body_as_str(self) -> str:
        """
        Return a string representation of the body of a paper.

        :return: A string containing the body text.
        :rtype: str
        """
        s = ""
        if self.body:
            for sec in self.body:
                s += "\n"
                s += str(sec)
        return s

    def __bool__(self):
        """
        Determine the truth value of a Paper object based on successful
        initialization.

        :return: True if the Paper object was successfully initialized with
            data, False otherwise.
        :rtype: bool
        """
        return self.has_data

    def full_text(self, print_text: bool = False):
        """
        Return the full abstract and/or body text of this Paper as a string.

        Optionally, you can choose to print the text.

        :param bool print_text: If True, print the text; if False, return it
            as a string.

        :return: A string containing the full text of the abstract and/or body.
        :rtype: str
        """
        s = ""
        if self.abstract:
            s += "Abstract: \n"
            s += self.abstract_as_str()
        if self.body:
            s += "Body: \n"
            s += self.body_as_str()

        if print_text:
            print(s)
        return s

    def __str__(self):
        """
        Return a string representation of the Paper object.

        :return: A string containing the PMCID, title, abstract, and body text
            of the paper.
        :rtype: str
        """
        s = ""
        s += f"\nPMCID: {self.pmcid}\n"
        s += f"Title: {self.title}\n"
        # Append all text from abstract PaperSections
        s += "\nAbstract:\n"
        if self.abstract:
            for sec in self.abstract:
                s += str(sec)
        # Append all text from body PaperSections
        s += "\nBody:\n"
        if self.body:
            for sec in self.body:
                s += str(sec)
        return s

    def __eq__(self, other):
        """
        Check if two Paper objects are equal.

        Two Paper objects are considered equal if they share the same PMCID and have
        the same date of last update. Papers with the same content but downloaded or
        parsed on different dates are not considered equal.

        To compare Paper objects based solely on their PMCID, use
        `Paper1.pmcid == Paper2.pmcid`.

        Note that articles that are not open access on PMC may not have a PMCID, and a
        unique comparison method will be needed for these cases. However, most papers
        downloaded via ScrapeMed should have a PMCID.

        :param other: The other Paper object to compare.
        :type other: Paper
        :return: True if the two Paper objects are equal, False otherwise.
        :rtype: bool
        """
        if not self:
            return False
        return self.pmcid == other.pmcid and self.last_updated == other.last_updated

    def to_relational(self) -> pd.Series:
        """
        Generate a pandas Series representation of the paper.

        This method creates a pandas Series containing a relational representation of
        the paper's data. Some data may be lost in this process, but most useful text
        data and metadata will be retained in a structured form.

        :return: A pandas Series representing the paper's data.
        :rtype: pd.Series
        """

        data = {
            "PMCID": self.pmcid,
            "Last_Updated": self.last_updated,
            "Title": self.title,
            "Authors": self._extract_names(self.authors)
            if isinstance(self.authors, pd.DataFrame)
            else None,
            "Non_Author_Contributors": self._extract_names(self.non_author_contributors)
            if isinstance(self.non_author_contributors, pd.DataFrame)
            else None,
            "Abstract": self.abstract_as_str(),
            "Body": self.body_as_str(),
            "Journal_ID": self.journal_id,
            "Journal_Title": self.journal_title,
            "ISSN": self.issn,
            "Publisher_Name": self.publisher_name,
            "Publisher_Location": self.publisher_location,
            "Article_ID": self.article_id,
            "Article_Types": self.article_types,
            "Article_Categories": self.article_categories,
            "Published_Date": self._serialize_dict(self.published_date)
            if isinstance(self.published_date, dict)
            else None,
            "Volume": self.volume,
            "Issue": self.issue,
            "First_Page": self.fpage,
            "Last_Page": self.lpage,
            "Copyright": self.copyright,
            "License": self.license,
            "Funding": self.funding,
            "Footnote": self.footnote,
            "Acknowledgements": self.acknowledgements,
            "Notes": self.notes,
            "Custom_Meta": self.custom_meta,
            "Ref_Map": self.ref_map,
            "Citations": [
                self._serialize_dict(c) for c in self.citations if isinstance(c, dict)
            ],
            "Tables": [
                self._serialize_df(t)
                for t in self.tables
                if isinstance(t, (pd.io.formats.style.Styler, pd.DataFrame))
            ],
            "Figures": self.figures,
        }
        return pd.Series(data)

    # ---------------Helper functions for to_relational---------------------
    def _extract_names(self, df):
        """
        Extract and format names from a DataFrame.

        :param df: The DataFrame containing name data.
        :type df: pd.DataFrame
        :return: A list of formatted names.
        :rtype: List[str]
        """
        return df.apply(
            lambda row: f"{row['First_Name']} {row['Last_Name']}", axis=1
        ).tolist()

    def _serialize_dict(self, data_dict):
        """
        Serialize a dictionary into a string.

        :param data_dict: The dictionary to serialize.
        :type data_dict: dict
        :return: A string representation of the serialized dictionary.
        :rtype: str
        """
        return "; ".join([f"{key}: {value}" for key, value in data_dict.items()])

    def _serialize_df(self, df):
        """
        Serialize a DataFrame into an HTML string.

        :param df: The DataFrame to serialize.
        :type df: pd.DataFrame
        :return: An HTML representation of the serialized DataFrame.
        :rtype: str
        """

        return df.to_html()

    # ---------------End Helper functions for to_relational--------------------

    def vectorize(
        self, chunk_size: int = 100, chunk_overlap: int = 20, refresh: bool = False
    ):
        """
        Generate an in-memory vector database representation of the paper.

        This method generates an in-memory vector database representation of the
        paper, stored in `paper.vector_collection`. It focuses on vectorizing the
        abstract and body text.

        :param int chunk_size: An approximate chunk size to split the paper into
            (measured in characters).
        :param int chunk_overlap: An approximate desired chunk overlap
            (measured in characters).
        :param bool refresh: Whether or not to clear and re-vectorize the paper
            with new settings.

        :return: None
        """
        if not refresh and self.vector_collection:
            print(
                (
                    "Paper already vectorized! To re-vectorize with new "
                    "settings, pass refresh=True."
                )
            )
            return None

        print("Vectorizing Paper (This may take a little while)...")
        if len(self.full_text()) == 0:
            warnings.warn(
                "Attempted to vectorize a Paper with no text. Aborting.",
                emptyTextWarning,
            )
            return None

        # Set up an in-memory chromadb collection for this paper
        client = chromadb.Client()
        try:
            self.vector_collection = client.get_or_create_collection(
                f"Paper-PMCID-{self.pmcid}"
            )
        except AttributeError:
            self.vector_collection = client.get_or_create_collection(
                f"Paper-Random-UUID-{uuid.uuid4()}"
            )

        # setup chunk model
        chunk_model = CharacterTextSplitter(
            separator="\\n\\n|\\n|\\.|\\s",
            is_separator_regex=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            keep_separator=True,
        )

        # chunk the text, add metadata for the PMCID each chunk
        # originates from, add unique chunk ids
        p_chunks = chunk_model.split_text(self.full_text())
        p_metadatas = [{"pmcid": self.pmcid}] * len(p_chunks)
        try:
            pmcid = self.pmcid
        except AttributeError:
            pmcid = uuid.uuid4()
        p_ids = [self._generate_chunk_id(pmcid, i) for i in range(len(p_chunks))]

        # upload the chunked texts into the vector collection
        self.vector_collection.add(documents=p_chunks, metadatas=p_metadatas, ids=p_ids)

        print(
            (
                "Done Vectorizing Paper! Natural language query with "
                "Paper.query() now available."
            )
        )
        return None

    # -----------------helper funcs for self.vectorize-----------------
    def _generate_chunk_id(self, pmcid: str, index: Union[int, str]):
        """
        Generate an ID for a PMC text chunk using the PMCID and the chunk's index.

        The chunk indices should be unique. It is recommended to use indexes from
        the result of the chunk model.

        :param str pmcid: The PMCID of the paper.
        :param Union[int, str] index: The index of the chunk.
        :return: A unique chunk ID.
        :rtype: str
        """
        return f"pmcid-{pmcid}-chunk-{str(index)}"

    def _get_chunk_index_from_chunk_id(self, chunk_id: str) -> str:
        """
        Given a PMCID Chunk ID in the format generated by `_generate_chunk_id`,
        extract the index of the chunk.

        :param str chunk_id: The chunk ID.
        :return: The index of the chunk.
        :rtype: str
        """
        pattern = re.compile(r"chunk-(\d+)")  # Compile the regex pattern
        match = pattern.search(chunk_id)
        index = None
        if match:
            index = match.group(1)
        return index

    def _get_pmcid_from_chunk_id(self, chunk_id: str) -> str:
        """
        Given a PMCID Chunk ID in the format generated by `_generate_chunk_id`,
        extract the PMCID of the chunk.

        :param str chunk_id: The chunk ID.
        :return: The PMCID of the chunk.
        :rtype: str
        """
        pattern = re.compile(r"pmcid-(\d+)")  # Compile the regex pattern
        match = pattern.search(chunk_id)
        pmcid = None
        if match:
            pmcid = match.group(1)
        return pmcid

    # -----------------end helper funcs for self.vectorize-----------------

    def query(
        self, query: str, n_results: int = 1, n_before: int = 2, n_after: int = 2
    ) -> Dict[str, str]:
        """
        Query the paper with natural language questions.

        :param str query: The natural language question/query.
        :param int n_results: The number of most semantically similar paper
            sections to retrieve.
        :param int n_before: The number of chunks before the match to include
            in the combined output.
        :param int n_after: The number of chunks after the match to include in
            the combined output.

        :return: A dictionary with keys representing the most semantically
            similar result chunk(s) and values representing the paper text(s)
            around the most semantically similar result chunk(s).
            The text length is determined by the chunk size used in
            `self.vectorize()` and the params `n_before` and `n_after`.
        :rtype: dict[str, str]
        """

        result = self.expanded_query(
            query=query, n_results=n_results, n_before=n_before, n_after=n_after
        )

        return result

    # -----------------helper funcs for self.query----------------------
    def expanded_query(
        self, query: str, n_results: int = 1, n_before: int = 2, n_after: int = 2
    ) -> Dict[str, str]:
        """
        Query the paper with an expanded natural language question/query.

        This method matches a natural language query with the vectorized Paper.
        It retrieves and expands the text sections around the most semantically
        similar result chunk(s).

        :param str query: The natural language query.
        :param int n_results: The number of most semantically similar paper
            sections to retrieve.
        :param int n_before: The number of chunks before the match to include
            in the combined output.
        :param int n_after: The number of chunks after the match to include
            in the combined output.

        :return: A dictionary with keys representing the most semantically
            similar result chunk(s) and values representing the expanded paper
            text(s) around the result chunk(s).
        :rtype: dict[str, str]
        """
        # if the paper has not already been vectorized, vectorize
        if not self.vector_collection:
            self.vectorize()
        # if vectorization fails, abort
        if not self.vector_collection:
            return None

        result = self.vector_collection.query(
            query_texts=[query], include=["documents"], n_results=n_results
        )

        expanded_results = {}
        for id in result["ids"][0]:
            chunk_index = self._get_chunk_index_from_chunk_id(id)
            pmcid = self._get_pmcid_from_chunk_id(id)
            # get the texts before and after the result chunk
            expanded_ids = []
            for i in range(1, n_before + 1):
                expanded_ids.append(
                    self._generate_chunk_id(pmcid, int(chunk_index) - i)
                )
            expanded_ids.append(id)
            for i in range(1, n_after + 1):
                expanded_ids.append(
                    self._generate_chunk_id(pmcid, int(chunk_index) + i)
                )

            expanded_results[f"Match on {id}"] = self.vector_collection.get(
                ids=expanded_ids,
            )["documents"]

        cleaned_results = {}
        # append docs together two at a time, removing overlap
        for match, docs in expanded_results.items():
            combined_result = ""
            # combined docs together
            if len(docs) == 0:
                combined_result = None
            elif len(docs) == 1:
                combined_result = docs[0]
            else:
                # combine first two docs, removing overlap, to
                # start the combined result
                substring_match = SequenceMatcher(
                    None, docs[0], docs[1]
                ).find_longest_match(0, len(docs[0]), 0, len(docs[1]))
                combined_docs = (
                    docs[0][: substring_match.a] + docs[1][substring_match.b :]
                )
                combined_result += combined_docs
                # eat these first two docs
                if len(docs) >= 3:
                    docs = docs[2:]
                else:
                    docs = []
                # continue eating the rest one by one
                while len(docs) >= 1:
                    substring_match = SequenceMatcher(None, combined_result, docs[0])

                    substring_match = substring_match.find_longest_match(
                        0, len(combined_result), 0, len(docs[0])
                    )

                    combined_result = (
                        combined_result[: substring_match.a]
                        + docs[0][substring_match.b :]
                    )

                    # eat the processed doc
                    if len(docs) >= 2:
                        docs = docs[1:]
                    else:
                        docs = []

                cleaned_results[match] = "..." + combined_result + "..."

        return cleaned_results


# --------------------END PAPER OBJECT SCHEMA-------------------------------
