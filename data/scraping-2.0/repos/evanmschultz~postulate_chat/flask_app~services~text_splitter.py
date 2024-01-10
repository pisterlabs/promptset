from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    """
    Responsible for splitting text within the documents into manageable chunks.
    """

    def split_text(self, documents: list) -> list:
        """
        Splits the list of documents into smaller text chunks.

        Args:
            documents (list): The list of documents to split.

        Returns:
            list: The list of split text chunks.
        """
        print("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "(?<=\\. )", " ", ""],
        )
        split_documents: list = text_splitter.split_documents(documents)
        print(
            f"\n{'_'*80}\n\nSplit documents into {len(split_documents)} chunks.\n\n{'_'*80}"
        )
        return split_documents
