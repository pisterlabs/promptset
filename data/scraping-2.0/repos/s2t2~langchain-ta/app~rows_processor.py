
from functools import cached_property

from langchain.document_loaders import DataFrameLoader

from app.cell import Cell
from app.document_processor import DocumentProcessor, CHUNK_OVERLAP, CHUNK_SIZE, SIMILARITY_THRESHOLD


# hacky class for allowing us to process documents from a number of rows
# ... instead of reading from a given filepath
# todo: refactor and use mixins maybe
class RowsDocumentProcessor(DocumentProcessor):
    """Processes a collection of row documents."""

    #def __init__(self, rows_df, filepath, chunk_overlap=CHUNK_OVERLAP, chunk_size=CHUNK_SIZE, verbose=True, similarity_threshold=SIMILARITY_THRESHOLD, file_id=None):
    #    super().__init__(filepath=filepath, chunk_overlap=chunk_overlap, chunk_size=chunk_size, verbose=verbose, similarity_threshold=similarity_threshold, file_id=file_id)
    #    self.rows_df = rows_df.copy()
    #    print("ROWS:", len(self.rows_df))

    def __init__(self, rows_df, chunk_overlap=CHUNK_OVERLAP, chunk_size=CHUNK_SIZE, verbose=True, similarity_threshold=SIMILARITY_THRESHOLD):

        self.rows_df = rows_df.copy()
        self.filename = rows_df["filename"].unique()[0] # take the first, they should all be the same
        self.file_id = rows_df["file_id"].unique()[0] # take the first, they should all be the same

        self.chunk_overlap = int(chunk_overlap)
        self.chunk_size = int(chunk_size)

        self.embeddings_model_name = "text-embedding-ada-002"
        #self.faiss_index = self.filepath.upper().replace(".IPYNB", "") + "_FAISS_INDEX"
        self.similarity_threshold = float(similarity_threshold)

        self.verbose = bool(verbose)
        if self.verbose:
            print("---------------------")
            print("FILENAME:", self.filename)
            print("ROWS:", len(self.rows_df))


    # OVERWRITE PARENT METHODS WE DON'T NEED

    @cached_property
    def docs(self):
        return []

    @cached_property
    def doc(self):
        return None

    # OVERWRITE PARENT METHOD TO GET CELLS STRAIGHT FROM THE ROWS DATAFRAME:

    @cached_property
    def cells(self):
        loader = DataFrameLoader(self.rows_df, page_content_column="page_content")
        docs = loader.load()
        # wrap docs in cell class, to stay consistent with parent method
        docs = [Cell(page_content=doc.page_content, metadata=doc.metadata) for doc in docs]
        return docs # cell_docs
