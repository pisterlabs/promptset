import os
import marqo
from langchain.vectorstores import Marqo


class IndexRepository:
    def __init__(self, indexName=None):
        self.client = marqo.Client(os.getenv("MARQO_URL"))
        self.index_name = indexName
        self.indexes = self.__indexName()

    def createIndex(self, indexName=None):
        if indexName is None:
            indexName = self.index_name

        if indexName is not None and self.index_name not in self.indexes:
            print("Creating Index {}".format(indexName))
            return self.client.create_index(indexName)

        return "Index Name not defined"

    def deleteIndex(self, indexName=None):
        if indexName is None:
            indexName = self.index_name

        if indexName is not None and indexName in self.indexes:
            print("Deleting Index")
            self.client.delete_index(indexName)

            return "Index {} was deleted successfully".format(indexName)

        return "Failed to delete {}".format(indexName)

    def listIndex(self):
        return self.indexes

    def getAllItems(self, query, indexName=None):
        if indexName is None:
            indexName = self.index_name

        if indexName is not None:
            result = self.client.index(indexName).search(
                q=query,
                limit=5,
                show_highlights=True,
                filter_string="*:*",
                search_method=marqo.SearchMethods.LEXICAL,
            )

            return result

        return "No indexName defined"

    def vestorstore(self, indexName=None):
        if indexName is None:
            indexName = self.index_name

        if indexName is not None:
            return Marqo(self.client, indexName)

    def deleteDocument(self, ids, indexName=None):
        if indexName is None:
            indexName = self.index_name

        if indexName is not None:
            try:
                return self.client.index(indexName).delete_documents(ids=ids)
            except Exception:
                print("file Ids {} not found".format(ids))

    def getdocumentIds(self, fileHash, indexName=None):
        ids = []
        if indexName is None:
            indexName = self.index_name

        if indexName is not None:
            items = self.client.index(indexName).get_document(
                document_id=fileHash
            )
            ids = items['documents']

        return ids

    def __indexName(self):
        indexNames = []
        for indexName in self.client.get_indexes()['results']:
            indexNames.append(indexName.index_name)

        return indexNames
