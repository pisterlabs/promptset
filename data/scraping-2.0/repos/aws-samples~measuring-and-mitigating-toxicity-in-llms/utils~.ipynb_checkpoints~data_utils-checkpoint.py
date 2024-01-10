import pandas as pd
import ast
from convokit import Corpus, download
import gc 
from langchain.schema import Document

def _prepare_data():
    """
    
    """
    # download data
    corpus = Corpus(filename=download("movie-corpus", verbose=False))

    # obtain keys for all dialogs across various movies
    utter_keys = list(corpus.utterances.keys())
    convo_keys = list(corpus.conversations.keys())

    # initialize dataframe
    movie_df = pd.DataFrame(columns=["movie", "dialogue"])

    # create empty list to store movie name, dialogue
    movie_ls = []
    text_ls = []
    genre_dict = dict()

    # loop through all utterances and append to list
    for u in utter_keys:
        movie_ls.append(corpus.utterances[u].speaker.meta["movie_name"])
        text_ls.append(corpus.utterances[u].text)

    # loop through conversations and append to dictionary
    for c in convo_keys:
        try:
            genre_dict[corpus.conversations[c].meta["movie_name"]] = ast.literal_eval(corpus.conversations[c].meta["genre"])[0]
        except:
            genre_dict[corpus.conversations[c].meta["movie_name"]] = "none"
    # fill dataframe with data
    movie_df["movie"] = movie_ls
    movie_df["dialogue"] = text_ls

    # group by movie title and concatenate all text into one long dialogue
    grouped_df = (
        movie_df.groupby("movie")["dialogue"].apply(lambda x: " ".join(x)).reset_index()
    )

    # join with genre data
    grouped_df = grouped_df.merge(pd.DataFrame({"movie": genre_dict.keys(), "genre": genre_dict.values()}), on="movie")
    
    # delete objects that are no longer in use
    del corpus, utter_keys, convo_keys, movie_df, movie_ls, text_ls, genre_dict
    
    # garbage collect
    gc.collect()
    
    return grouped_df

def _add_summaries(sample, chain):
    """
    Function to create summaries of the movie dialogue dataset.
    """
    # turn off verbosity for chain
    chain.llm_chain.verbose = False

    # create LangChain document from the chunks
    docs = [
        Document(page_content=split["text"], metadata=split["metadata"])
        for split in sample["chunks"]
    ]

    # parse documents through the map reduce chain
    full_output = chain({"input_documents": docs})
    
    # extract the summary
    summary = full_output["output_text"]
    
    # return the new column
    sample["summary"] = summary
    
    # delete objects that are no longer in use
    del docs, summary
    
    # garbage collect
    gc.collect()
    
    return sample