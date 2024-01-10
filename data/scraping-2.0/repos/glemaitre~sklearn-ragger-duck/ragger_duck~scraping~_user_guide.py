"""Utilities to scrape User Guide documentation."""
import logging
import re
from itertools import chain
from numbers import Integral
from pathlib import Path

from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import Interval

from ._shared import _chunk_document

SKLEARN_USER_GUIDE_URL = "https://scikit-learn.org/stable/modules/"
loogger = logging.getLogger(__name__)


def _user_guide_path_to_user_guide_url(path):
    """Convert a User Guide path to an User Guide URL.

    Parameters
    ----------
    path : :class:`pathlib.Path`
        The path to the User Guide documentation.

    Returns
    -------
    str
        The User Guide URL.
    """
    return SKLEARN_USER_GUIDE_URL + path.name


def extract_user_guide_doc_from_single_file(html_file):
    """Extract the text from the User Guide documentation.

    This function can process classes and functions.

    Parameters
    ----------
    html_file : :class:`pathlib.Path`
        The path to the HTML User Guide documentation.

    Returns
    -------
    dict
        A dictionary containing the source and text of the User Guide documentation.
    """
    if not isinstance(html_file, Path):
        raise ValueError(
            "The User Guide HTML file should be a pathlib.Path object. "
            f"Got {html_file!r}."
        )
    if html_file.suffix != ".html":
        raise ValueError(
            f"The file {html_file} is not an HTML file. Please provide an HTML file."
        )
    with open(html_file, "r") as file:
        soup = BeautifulSoup(file, "html.parser")

    text = soup.get_text("")
    # Remove line breaks within a paragraph
    newline = re.compile(r"\n\s*")
    text = newline.sub(r"\n", text)
    # Remove the duplicated spaces on the fly
    multiple_spaces = re.compile(" +")
    text = multiple_spaces.sub(" ", text)

    return {
        "source": _user_guide_path_to_user_guide_url(html_file),
        "text": text,
    }


def _extract_user_guide_doc(user_guide_doc_folder):
    """Extract text from each HTML User Guide files from a folder

    Parameters
    ----------
    user_guide_doc_folder : :class:`pathlib.Path`
        The path to the User Guide documentation folder.

    Returns
    -------
    list of dict
        A list of dictionaries containing the source and text of the API
        documentation.
    """
    if not isinstance(user_guide_doc_folder, Path):
        raise ValueError(
            "The User Guide documentation folder should be a pathlib.Path object. Got "
            f"{user_guide_doc_folder!r}."
        )
    return [
        extract_user_guide_doc_from_single_file(html_file)
        for html_file in user_guide_doc_folder.glob("*.html")
    ]


class UserGuideDocExtractor(BaseEstimator, TransformerMixin):
    """Extract text from the User Guide documentation.

    This function can process classes and functions.

    Parameters
    ----------
    chunk_size : int or None, default=300
        The size of the chunks to split the text into. If None, the text is not chunked.

    chunk_overlap : int, default=50
        The overlap between two consecutive chunks.

    n_jobs : int, default=None
        The number of jobs to run in parallel. If None, then the number of jobs is set
        to the number of CPU cores.

    Attributes
    ----------
    text_splitter_ : :class:`langchain.text_splitter.RecursiveCharacterTextSplitter`
        The text splitter to use to chunk the document. If `chunk_size` is None, this
        attribute is None.
    """

    _parameter_constraints = {
        "chunk_size": [Interval(Integral, left=1, right=None, closed="left"), None],
        "chunk_overlap": [Interval(Integral, left=0, right=None, closed="left")],
        "n_jobs": [Integral, None],
    }

    def __init__(self, *, chunk_size=300, chunk_overlap=50, n_jobs=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.n_jobs = n_jobs

    def fit(self, X=None, y=None):
        """No-op operation, only validate parameters.

        Parameters
        ----------
        X : None
            This parameter is ignored.

        y : None
            This parameter is ignored.

        Returns
        -------
        self
            The fitted estimator.
        """
        self._validate_params()
        if self.chunk_size is not None:
            self.text_splitter_ = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " "],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
        else:
            self.text_splitter_ = None
        return self

    def transform(self, X):
        """Extract text from the API documentation.

        Parameters
        ----------
        X : :class:`pathlib.Path`
            The path to the API documentation folder.

        Returns
        -------
        output : list
            A list of dictionaries containing the source and text of the User Guide
            documentation.
        """
        if self.chunk_size is None:
            output = _extract_user_guide_doc(X)
        else:
            output = list(
                chain.from_iterable(
                    Parallel(n_jobs=self.n_jobs, return_as="generator")(
                        delayed(_chunk_document)(self.text_splitter_, document)
                        for document in _extract_user_guide_doc(X)
                    )
                )
            )
        if not output:
            raise ValueError(
                "No User Guide documentation was extracted. Please check the "
                "input folder."
            )
        return output

    def _more_tags(self):
        return {"X_types": ["string"], "stateless": True}
