import getpass
import html
import json
import os
import re
import shutil
from pathlib import Path
from posixpath import basename
from urllib.parse import urlparse

import chromadb
import numpy as np
import openai
import pandas as pd
import wikipedia
import wptools
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.llms import OpenAI
from llama_index.schema import TextNode
from llama_index.vector_stores import ChromaVectorStore
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


MAINFOLDER = Path(__file__).parent
DATAPATH = MAINFOLDER / "movie.parquet"
# Movie table database with
# "release_year",
# "title",
# "director",
# "cast",
# "genre",
# "wiki_page",
# "plot",
PERSIST_DIR = MAINFOLDER / "data"


def extract_number(text, indicator=None):
    """
    Return the relevant number as scrapped from wikipedia infoboxes.
    
    Parameters
    ----------
    text : str
        Text to extract number from.
    indicator : str, optional
        Indicator to find number e.g. $, by default None.
        
    Returns
    -------
    float
        Number extracted from text.
    """
    if "citation" in text:
        text = text.split("citation")[0]

    pattern = r"\s"
    text = re.sub(pattern, "", text)
    text = text.split("-")[-1]

    if indicator is not None:
        textsplit = text.split(indicator)
        if textsplit[-1] and textsplit[-1][0].isnumeric():
            match = re.findall(r"(\d+(?:\.\d+)?)", textsplit[-1])
            if match:
                return float(match[-1])
            match = re.findall(r"(\d+)", textsplit[-1])
            if match:
                return float(match[-1])

    match = re.findall(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match[-1])

    match = re.findall(r"(\d+)", text)
    if match:
        return float(match[-1])

    return np.nan


def convert_text_amount_to_number(text):
    """
    Convert text amount of money to number e.g. $1.2 million -> 1200000.
    
    Parameters
    ----------
    text : str
        Text to convert.
        
    Returns
    -------
    float
        Number extracted from text.
    """
    text = text.lower()
    # Remove commas
    text = text.replace(",", "")

    # Merge number word combinations
    text = re.sub(r"(\d+(\.\d+)?)([a-zA-Z]+)", r"\1 \3", text)

    # Convert abbreviations
    conversions = {
        "thousand": "000",
        "million": "000000",
        "billion": "000000000",
        "thous": "000",
        "mill": "000000",
        "bill": "000000000",
        "th": "000",
        "mil": "000000",
        "bil": "000000000",
        "k": "000",
        "m": "000000",
        "b": "000000000",
    }
    for abbv, replacement in conversions.items():
        if abbv in text:
            if "." in text:
                split = text.split(".")
                # add n zeros to split[1] after the initial digits
                if split[1]:
                    for idx, s in enumerate(split[1]):
                        if not s.isnumeric():
                            break
                    if idx > 3:
                        pass
                    else:
                        split[1] = split[1][:idx] + "0" * (3 - idx) + split[1][idx:]
                        text = split[0] + split[1]
                        replacement = replacement[3:]

            text = text.replace(abbv, replacement)
            break

    return extract_number(text, indicator="$")


def convert_time_to_minutes(text):
    """
    Convert text time to minutes e.g. 1 hour -> 60.
    
    Parameters
    ----------
    text : str
        Text to convert.
        
    Returns
    -------
    float
        Number extracted from text.
    """
    text = text.lower()
    # Remove commas
    text = text.replace(",", "")

    # Merge number word combinations
    text = re.sub(r"(\d+(\.\d+)?)([a-zA-Z]+)", r"\1 \3", text)

    if any([x in text for x in ["hour", "hr", "h"]]):
        return extract_number(text) * 60

    elif any([x in text for x in ["sec"]]):
        return extract_number(text) / 60

    return extract_number(text)


def clean_string(string):
    """
    Initial clean of string from wikipedia infoboxes.
    
    Parameters
    ----------
    string : str
        String to clean.
        
    Returns
    -------
    str
        Cleaned string.
    """
    string = re.sub(r"{{\w+}}", "", string)
    string = html.unescape(string)
    return string


def load_openai_api_key():
    """
    Load OpenAI API key from environment variable or prompt user to enter it.
    """
    if "OPENAI_API_KEY" not in os.environ or os.environ["OPENAI_API_KEY"] == "":
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
    openai.api_key = os.environ["OPENAI_API_KEY"]
    return os.environ["OPENAI_API_KEY"]


def _deal_with_wiki_page(page, year):
    # some films in the dataset might have slightly outdated urls and therefore title pages
    # Duplicate films with the same title and but different year might exist
    if not page.strip().endswith(f"({year})") or not page.strip().endswith(
        f"({year} film)"
    ):
        if page.strip().endswith("(film)"):
            page = page.replace("(film)", f"({year} film)")
        else:
            page = f"{page} ({year} film)"
    return page


def load_relevant_wiki_info(wiki_url, title, year):
    """
    Load relevant information from Wikipedia page.
    
    Parameters
    ----------
    wiki_url : str
        Wikipedia url.
    title : str
        Movie title.
    year : str
        Movie release year.
        
    Returns
    -------
    dict
        Dictionary of relevant information.
    """
    page = basename(urlparse(wiki_url).path)
    page = page.replace("_", " ")

    # catch wikipedia.exceptions.PageError
    try:
        # first try to use the year and title to find the page
        try:
            wiki = wikipedia.page(f"{title} {year} movie")
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            try:
                wiki = wikipedia.page(page)
            except wikipedia.exceptions.PageError:
                page = title
                wiki = wikipedia.page(page)
            except wikipedia.exceptions.DisambiguationError:
                page = _deal_with_wiki_page(page, year)
                wiki = wikipedia.page(page)
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
        return {"title": title, "categories": []}

    categories = getattr(wiki, "categories", [])
    title = getattr(wiki, "title", "")
    wpage = wptools.page(wiki.title).get_parse()
    if wpage.data["infobox"] is None:
        return {"title": title, "categories": categories}

    infos = wpage.data["infobox"]
    infos["title"] = title
    infos["categories"] = categories
    return infos


class DataTools:
    """
    Tool to load data and add it to the database, as well 
    as retrieve movies from the database using natural language.
    
    Parameters
    ----------
    datapath : pathlib.Path, optional
        Path to parquet file containing movie descriptions, by default DATAPATH.
    persist_dir : pathlib.Path, optional
        Path to directory to store database, by default PERSIST_DIR.
    overwrite : bool, optional
        Whether to overwrite existing database, by default False.
    metadata_filter : str, optional
        Metadata filter to use, by default "fast".
        If "fast" then use regex to remove nuisance characters.
        If "accurate" then use GPT-3.5 to reformat the metadata.
    add_theme : bool, optional
        Whether to add central theme to metadata, by default False.
        If True then use GPT-3.5 to infer the central theme of the movie.
    model_name : str, optional
        Name of OpenAI model to use for retrieval, by default "gpt-4".
    """

    def __init__(
        self,
        datapath=DATAPATH,
        persist_dir=PERSIST_DIR,
        overwrite=False,
        metadata_filter="fast",  # "fast" or "accurate"
        add_theme=False,
        model_name="gpt-4",
    ):
        self.model_name = model_name
        self.datapath = datapath
        self.persist_dir = persist_dir
        self.overwrite = overwrite
        self.metadata_filter = metadata_filter
        self.add_theme = add_theme

        load_openai_api_key()
        self.llm = OpenAI(temperature=0, model=model_name)
        self.service_context = ServiceContext.from_defaults(llm=self.llm)

        self.set_vector_store()
        self.set_metadata_string_formatter()
        self.set_central_theme_chain()
        self.set_medata_df()

    def set_central_theme_chain(self):
        """
        Set theme chain to infer central theme of movie.
        """
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        central_theme_prompt = PromptTemplate(
            input_variables=["title", "plot"],
            template=(
                "Please write a comma-separated list of the"
                'central themes of the movie "{title}" with the following plot: \n'
                "# Plot:\n"
                "{plot}"
            ),
        )
        chain = LLMChain(llm=llm, prompt=central_theme_prompt)
        self.central_theme_chain = chain
        return self

    def set_metadata_string_formatter(self):
        """
        Set metadata string formatter
        """
        if self.metadata_filter == "fast":
            self.chars_to_remove = re.compile(r"(\[\[)|\\\n|(\{\{)|(Plainlist)")
        elif self.metadata_filter == "accurate":
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            format_prompt = PromptTemplate(
                input_variables=["context", "text"],
                template=(
                    "Please reformat the list below of people, companies, or things "
                    "involved in a movie for {context} "
                    "into a clean comma-separated list "
                    "with all links and "
                    "non-alphanumeric characters removed:\n"
                    "{text}"
                ),
            )
            chain = LLMChain(llm=llm, prompt=format_prompt)
            self.metadata_string_formatter = chain
        else:
            raise ValueError("metadata_filter must be 'fast' or 'accurate'")
        return self

    def set_medata_df(self):
        """
        Set metadata dataframe for filtering and retrieval.
        Clean various columns, convert their data types if relevant,
        and set them to lower case.
        """
        metadata = self.chroma_collection.get()["metadatas"]
        if not metadata:
            return self
        metadata = pd.DataFrame(metadata)
        metadata["id_"] = metadata["_node_content"].apply(
            lambda x: json.loads(x)["id_"]
        )
        metadata.drop(columns=["_node_content"], inplace=True)
        # set all string columns to lower case
        metadata = metadata.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        # numeric columns : budget, gross, runtime, release_year
        for numeric_column in ["budget", "gross"]:
            metadata[numeric_column] = metadata[numeric_column].apply(convert_text_amount_to_number)
        metadata["runtime"] = metadata["runtime"].apply(convert_time_to_minutes)
        self.metadata_df = metadata
        return self

    def update_metadata_df(self):
        """
        Update metadata dataframe for filtering and retrieval.
        """
        return self.set_medata_df()

    def format_metadata_string(self, context, text):
        """
        Format metadata string for storage
        """
        if self.metadata_filter == "fast":
            return re.sub(self.chars_to_remove, "", text)
        else:
            return self.metadata_string_formatter.run(
                {"context": context, "text": text}
            )

    def get_data_and_add_similarities(self):
        """
        Load data from parquet file and add similar movies to metadata using LDA.
        """
        data = pd.read_parquet(self.datapath).fillna("")
        self.data = data
        assert set(
            [
                "release_year",
                "title",
                "director",
                "cast",
                "genre",
                "wiki_page",
                "plot",
            ]
        ).issubset(set(data.columns))

        # drop duplicate titles
        data = data.drop_duplicates(subset=["title"])

        # Use LDA to get a reduced representation of the plot text
        # Use this to find similar movies and add this to the metadata
        vectorizer = TfidfVectorizer(
            max_df=0.95, min_df=2, max_features=1000, stop_words="english"
        )
        vectorized_words = vectorizer.fit_transform(data["plot"])
        lda = LDA(
            n_components=10, max_iter=7, learning_method="online", learning_offset=50.0
        )
        Xt = lda.fit_transform(vectorized_words)
        dist = pairwise_distances(Xt)
        # five most similar movies according to their topics
        k = 5
        argsort = np.argsort(dist, axis=1)[:, 1 : k + 1]
        similar_movies = data["title"].to_numpy()[argsort]
        # make it a comma separated list
        similar_movies = [", ".join(similar) for similar in similar_movies]
        # add to data
        data["similar_movies"] = similar_movies

        return data

    def set_vector_store(self):
        """
        Load vector store index from database or create new one.
        """
        if self.overwrite and os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

        chroma_client = chromadb.Client(
            settings=chromadb.Settings(
                is_persistent=True, persist_directory=str(self.persist_dir / "chromadb")
            )
        )
        chroma_collection = chroma_client.get_or_create_collection("movie")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.persist(persist_dir=self.persist_dir)

        self.chroma_client = chroma_client
        self.chroma_collection = chroma_collection
        self.vector_store = vector_store
        self.storage_context = storage_context
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            show_progress=True,
        )

        return self

    def process_info(self, info, key):
        """
        Process info from wikipedia infoboxes.
        """
        if key in info:
            if re.search(r"(\[\[)|\\\n|(\{\{)|(Plainlist)", info[key]) is not None:
                return self.format_metadata_string(key, info[key])
            else:
                return clean_string(info[key])
        else:
            return ""

    def get_movie_node(self, row):
        """
        Get movie node from row of dataframe.
        Obtain metadata from wikipedia page.
        """
        # row must have keys:
        # - plot
        # - title
        # - release_year
        # - director
        # - genre
        # - cast
        # - wiki_page
        if not isinstance(row, pd.Series):
            row = pd.Series(row)

        row = row.fillna("")
        node_name = basename(urlparse(row["wiki_page"]).path)

        if self.chroma_collection.get(node_name)["ids"]:
            return

        text = row["plot"]
        metadata = {
            "title": row["title"],
            "release_year": row["release_year"],
            "director": row["director"],
            "genre": row["genre"],
            "cast": row["cast"],
            "similar_movies": row["similar_movies"],
        }
        if self.add_theme:
            metadata["theme"] = self.central_theme_chain.run(
                {"title": row["title"], "plot": row["plot"]}
            )
        # Extra features extracted from wikipedia:
        if row["wiki_page"]:
            infos = load_relevant_wiki_info(
                row["wiki_page"], row["title"], row["release_year"]
            )
            metadata["country"] = self.process_info(infos, "country")
            metadata["budget"] = self.process_info(infos, "budget")
            metadata["gross"] = self.process_info(infos, "gross")
            metadata["language"] = self.process_info(infos, "language")
            metadata["studio"] = self.process_info(infos, "studio")
            metadata["cinematography"] = self.process_info(infos, "cinematography")
            metadata["producer"] = self.process_info(infos, "producer")
            metadata["writer"] = self.process_info(infos, "writer")
            metadata["editing"] = self.process_info(infos, "editing")
            metadata["music"] = self.process_info(infos, "music")
            metadata["runtime"] = self.process_info(infos, "runtime")
            metadata["categories"] = ", ".join(infos["categories"])

        return TextNode(text=text, metadata=metadata, id_=node_name)

    def get_and_insert_nodes(self, df: pd.DataFrame):
        """
        Process dataframe and get movie nodes. Insert nodes into database.
        """
        nodes = []
        for _, row in df.iterrows():
            node = self.get_movie_node(row)
            if node is not None:
                nodes.append(node)
        self.index.insert_nodes(nodes)
        return None

    def get_filter_function_properties(self):
        """
        Arguments for filter_movies function that will be passed
        to the OpenAI API.
        """

        p = {}

        p["release_year"] = {
            "type": "string",
            "description": "A comma-separated list (e.g. 1999, 2005, 2008) or a range formatted with a dash symbol - (e.g. 1999-2005) of specific release years requested; for smaller than queries precede the budget with a < and for larger than queries precede the budget with a >; if the user specifies decades (e.g. 90s) convert the string to a range of years (e.g. 1990-1999)",
        }
        p["director"] = {
            "type": "string",
            "description": "Comma separated list of directors; for NOT queries precede the director with a ~ and for OR queries separate with a |",
        }
        p["genre"] = {
            "type": "string",
            "description": "Comma separated list of genres; for NOT queries precede the genre with a ~ and for OR queries separate with a |",
        }
        p["cast"] = {
            "type": "string",
            "description": "Comma separated list of cast members; for NOT queries precede the cast member with a ~ and for OR queries separate with a |",
        }
        p["similar_movies"] = {
            "type": "string",
            "description": "Comma separated list of similar movies to movie title in query.",
        }
        p["country"] = {
            "type": "string",
            "description": "Comma separated list of countries; for NOT queries precede the country with a ~ and for OR queries separate with a |",
        }
        p["budget"] = {
            "type": "string",
            "description": "Movie budget range formatted with a dash symbol - and only numerical characters (e.g. 4000000-50000); for smaller than queries precede the budget with a < and for larger than queries precede the budget with a >",
        }
        p["gross"] = {
            "type": "string",
            "description": "Movie gross range formatted with a dash symbol - and only numerical characters (e.g. 4000000-50000); for smaller than queries precede the gross with a < and for larger than queries precede the gross with a >",
        }
        p["language"] = {
            "type": "string",
            "description": "Comma separated list of languages; for NOT queries precede the language with a ~ and for OR queries separate with a |",
        }
        p["studio"] = {
            "type": "string",
            "description": "Comma separated list of studios; for NOT queries precede the studio with a ~ and for OR queries separate with a |",
        }
        p["cinematography"] = {
            "type": "string",
            "description": "Comma separated list of cinematographers; for NOT queries precede the cinematographer with a ~ and for OR queries separate with a |",
        }
        p["producer"] = {
            "type": "string",
            "description": "Comma separated list of producers; for NOT queries precede the producer with a ~ and for OR queries separate with a |",
        }
        p["writer"] = {
            "type": "string",
            "description": "Comma separated list of writers; for NOT queries precede the writer with a ~ and for OR queries separate with a |",
        }
        p["editing"] = {
            "type": "string",
            "description": "Comma separated list of editors; for NOT queries precede the editor with a ~ and for OR queries separate with a |",
        }
        p["music"] = {
            "type": "string",
            "description": "Comma separated list of music composers; for NOT queries precede the music composer with a ~ and for OR queries separate with a |",
        }
        p["runtime"] = {
            "type": "string",
            "description": "Movie runtime range formatted with a dash symbol - and only numerical characters in minutes (e.g. 120-160); for smaller than queries precede the runtime with a < and for larger than queries precede the runtime with a >",
        }
        p["categories"] = {
            "type": "string",
            "description": "Comma separated list of categories; for NOT queries precede the category with a ~ and for OR queries separate with a |",
        }
        if self.add_theme:
            p["theme"] = {
                "type": "string",
                "description": "Comma separated list of themes; for NOT queries precede the theme with a ~ and for OR queries separate with a |",
            }
            p["plot"] = {
                "type": "string",
                "description": "Overall plot  of movie the user is looking for. You can also infer this parameter from the user query, if it is not provided explicitly. If you infer this, keep it as general as possible (e.g. if the user query is only 'Keanu Reeves' set this parameter to 'gun fight movies').",
            }
        else:
            p["plot"] = {
                "type": "string",
                "description": "Overall plot or theme of movie the user is looking for. You can also infer this parameter from the user query, if it is not provided explicitly. If you infer this, keep it as general as possible (e.g. if the user query is only 'Keanu Reeves' set this parameter to 'gun fight movies').",
            }

        p["topk"] = {
            "type": "integer",
            "description": "Number of movies the user is requesting. If the user does not specify this parameter, set it to 5.",
        }

        return p

    def get_filter_function_dict(self):
        """
        Formatted dictionary for function calling in OpenAI API.
        """
        p = self.get_filter_function_properties()
        description = "Function to filter movies by using plot information and/or metadata contained in the user query."
        name = "filter_movies"
        return {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": p,
            },
            "required": ["plot"],
        }

    def retrieve(self, query):
        """
        Retrieve/Recommend movies from database using natural language.
        """
        function_dict = self.get_filter_function_dict()
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a movie recomender bot. "
                        "You recommend movies according "
                        f"to a function called `{function_dict['name']}` "
                        "that accepts various parameters. "
                        "Make sure to correct any spelling "
                        "mistakes in the user query. "
                        "Try to use follow the parameterization and "
                        "infer as much from the user query assuming "
                        "popular preferences. "
                        "Don't ask for more information just guess away "
                        "with the plot/theme parameter."
                    ),
                },
                {"role": "user", "content": query},
            ],
            functions=[function_dict],
            function_call="auto",
        )
        response = response["choices"][0]

        if response["finish_reason"] == "function_call":
            parameters = json.loads(response["message"]["function_call"]["arguments"])
            return self.filter_movies(parameters)
        else:
            return str(response["message"]["content"])

    @staticmethod
    def filter_numeric(parameters, numeric_param, metadata):
        """
        Filter numeric parameters as returned by the OpenAI API.
        """
        d = parameters[numeric_param].strip()
        try:
            if d.startswith("<"):
                d = d[1:]
                metadata = metadata[metadata[numeric_param] < d]
            elif d.startswith(">"):
                d = d[1:]
                metadata = metadata[metadata[numeric_param] > d]
            elif "-" in d:
                d = d.split("-")
                d = [int(x) for x in d]
                metadata = metadata[
                    (metadata[numeric_param] >= d[0])
                    & (metadata[numeric_param] <= d[1])
                ]
            elif "," in d:
                d = d.split(",")
                d = [int(x) for x in d]
                metadata = metadata[metadata[numeric_param].isin(d)]
            else:
                d = int(d)
                metadata = metadata[metadata[numeric_param] == d]
        except ValueError:
            # ignore invalid numeric parameters
            pass
        return metadata

    @staticmethod
    def filter_string(parameters, string_param, metadata):
        """
        Filter string parameters as returned by the OpenAI API.
        """
        d = parameters[string_param].strip().split(",")
        for d_ in d:
            d_ = d_.strip().lower()
            if d_.startswith("~"):
                d_ = d_[1:]
                metadata = metadata[~metadata[string_param].str.contains(d_)]
            elif "|" in d_:
                d_ = d_.split("|")
                d_ = [x.strip() for x in d_]
                bools = [metadata[string_param].str.contains(d__) for d__ in d_]
                metadata = metadata[np.any(bools, axis=0)]
            else:
                metadata = metadata[metadata[string_param].str.contains(d_)]
        return metadata

    def filter_movies(self, parameters):
        """
        Function to filter movies by using plot information and/or metadata contained in the user query.
        
        Parameters
        ----------
        parameters : dict
            Dictionary of the function parameters as returned by the OpenAI API.
            
        Returns
        -------
        list
            List of movie titles.
        """
        metadata = self.metadata_df
        for numeric_param in ["budget", "gross", "runtime", "release_year"]:
            if numeric_param not in parameters:
                continue
            metadata = self.filter_numeric(parameters, numeric_param, metadata)

        for string_param in [
            "director",
            "genre",
            "cast",
            "country",
            "language",
            "studio",
            "cinematography",
            "producer",
            "writer",
            "editing",
            "music",
            "categories",
            "theme",
        ]:
            if string_param not in parameters:
                continue
            metadata = self.filter_string(parameters, string_param, metadata)

        similar_plot = ""
        if "similar_movies" in parameters:
            similar_movies = parameters["similar_movies"].split(",")
            similar_movies = [x.strip().lower() for x in similar_movies]

            titles = []
            for movie in similar_movies:
                if movie not in metadata["title"].tolist():
                    if not similar_plot:
                        similar_plot += " Movies with a similar plot to "
                    similar_plot += movie + ", "
                else:
                    titles.extend(
                        self.metadata_df["similar_movies"][
                            self.metadata_df["title"] == movie
                        ]
                        .iloc[0]
                        .split(", ")
                    )

            if titles:
                metadata = metadata[metadata["title"].isin(titles)]

        plot = parameters.get("plot", "")
        plot += similar_plot
        topk = parameters.get("topk", 5)

        if not plot:
            # randomly select topk movies
            if len(metadata) < topk:
                return metadata["title"].tolist()
            return metadata["title"].sample(topk).tolist()

        where = {}
        where["title"] = {"$in": metadata["title"].tolist()}

        retriever = VectorIndexRetriever(
            self.index,
            similarity_top_k=topk,
            vector_store_kwargs={
                "where": where,
            },
            service_context=self.service_context,
        )
        results = retriever.retrieve(plot)
        ids = [x.node.id_ for x in results]
        return self.metadata_df["title"][
            self.metadata_df["id_"].isin(ids)
        ].tolist()


def _batch_iter_df(df, batch_size):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i: i + batch_size]


def add_data_to_db(tools: DataTools):
    """
    Adds data from the dataframe to the Chroma database.
    """
    batch_size = 64
    data = tools.get_data_and_add_similarities()

    for df in _batch_iter_df(data, batch_size):
        tools.get_and_insert_nodes(df)


if __name__ == "__main__":
    tools = DataTools(overwrite=False)
    add_data_to_db(tools)
