import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os

from dotenv import load_dotenv
load_dotenv()

# client = chromadb.EphemeralClient()
client = chromadb.PersistentClient(path="./db")

openai_token = os.getenv('OPENAI_TOKEN')
openai_embed = OpenAIEmbeddingFunction(
    api_key=openai_token
)

trashCollection = client.get_or_create_collection(
    name="TrashMail",
    embedding_function=openai_embed
)

inboxCacheCollection = client.get_or_create_collection(
    name="InboxCache",
    embedding_function=openai_embed
)

urlContextCacheCollection = client.get_or_create_collection(
    name="UrlContextCache",
    embedding_function=openai_embed
)

import hashlib
import json
import threading
import math
import time

class SafeInterface():
    def __init__(self, collection, charCap=3500, batchSize=200, threaded=True):
        self.col = collection
        self.batchSize = batchSize
        self.threaded = threaded
        self.charCap = charCap

    def __str__(self):
        output = ""
        output += f"SafeInterface:\n"
        output += f"\tCollection: {self.col}\n"
        output += f"\tBatch size: {self.batchSize}\n"

        return output

    def _addBatch(self, i, threadCount, start, end, ids, docs, metas=[]):
        truncatedDocs = [doc[:self.charCap] for doc in docs[start:end]]
        embeds = self.col._embed(truncatedDocs)

        if metas:
            self.col.add(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
                embeddings=embeds
            )
        else:
            self.col.add(
                ids=ids[start:end],
                documents=docs[start:end],
                embeddings=embeds
            )

        print(f"Finished thread {i} / {threadCount}")

    def addInBatches(self, ids, docs, metas=[]):
        numItems = len(ids)
        threads = []
        threadNum = math.ceil(numItems / self.batchSize)

        iters = 0
        for start in range(0, numItems, self.batchSize):
            end = min(start + self.batchSize, numItems)

            thread = threading.Thread(target=self._addBatch, args=(iters+1, threadNum, start, end, ids, docs, metas))
            threads.append(thread)
            thread.start()
            iters += 1

        for thread in threads:
            thread.join()

        return len(threads)


    def add(self, ids, documents, metadatas=None):
        addStart = time.time()
        if type(ids) == str:
            ids = [ids]
            documents = [documents]
            metadatas=[metadatas]

        # remove duplicates by ids
        idset = set()
        newIds = []
        newDocs = []
        newMetas = []
        for idx in ids:
            if idx not in idset:
                idset.add(idx)
                newIds.append(idx)
                newDocs.append(documents[ids.index(idx)])
                if metadatas:
                    newMetas.append(metadatas[ids.index(idx)])
        ids = newIds
        documents = newDocs
        if metadatas:
            metadatas = newMetas


        existing = self.col.get(ids=ids, include=["metadatas", "documents"])
        newIds = list(set(ids) - set(existing['ids']))

        # Add new docs
        if len(newIds) > 0:
            newDocs = []
            newMetas = []
            if metadatas:
                for idx, doc, meta in zip(ids, documents, metadatas):
                    if idx in newIds:
                        newDocs.append(doc)
                        newMetas.append(meta)
            else:
                for idx, doc in zip(ids, documents):
                    if idx in newIds:
                        newDocs.append(doc)
            print(f"Inserting {len(newIds)} new documents")
            iters = self.addInBatches(newIds, newDocs, newMetas)
            print(f"Added in {iters} batches of size {self.batchSize}")

        # Get object hashes for existing entries
        newVals = {}
        if metadatas:
            for idx, doc, meta in zip(ids, documents, metadatas):
                if idx in existing:
                    docHash = hashlib.sha256(doc.encode()),
                    metaString = json.dumps(meta, sort_keys=True)
                    metaHash = hashlib.sha256(metaString),
                    newVals[idx] = {
                        'doc': doc,
                        'docHash': docHash,
                        'meta': meta,
                        'metaHash': metaHash
                    }
        else:
            for idx, doc in zip(ids, documents):
                if idx in existing:
                    docHash = hashlib.sha256(doc.encode()),
                    newVals[idx] = {
                        'doc': doc,
                        'docHash': docHash,
                        'meta': None,
                        'metaHash': None
                    }

        # Find docs that don't match existing entries
        updatedIds = []
        updatedDocs = []
        updatedMetas = []
        for idx, doc, meta in zip(existing['ids'], existing['documents'], existing['metadatas']):
            existingDocHash = hashlib.sha256(doc.encode())
            metaString = json.dumps(meta, sort_keys=True)
            existingMetaHash = hashlib.sha256(metaString.encode())

            newData = newVals.get(idx, None)
            if newData:
                docMatch = existingDocHash != newData['docHash']
                metaMatch = existingMetaHash != newData['metaHash']

                if not docMatch or not metaMatch:
                    updatedIds.append(idx)
                    updatedDocs.append(newData['doc'])
                    updatedMetas.append(newData['meta'])

        # Update new changed entries
        if len(updatedIds) > 0:
            print(f"Updating {len(updatedIds)} documents")
            iters = self.addInBatches(updatedIds, updatedDocs, updatedMetas)
            print(f"Updated in {iters} batches of size {self.batchSize}")

        ignored = len(ids) - len(newIds) - len(updatedIds)
        if ignored != 0:
            print(f"Ignored {ignored} / {len(ids)} documents because they already exist in the collection")

        print(f"Interface finished add process in {time.time() - addStart} seconds")


if __name__ == '__main__':
    col = client.create_collection(name="test", embedding_function=openai_embed)

    inf = SafeInterface(col)

    print(f"Collection has: {col.count()} docs")
    inf.add(
        ids='1',
        documents='first document'
    )

    print(f"Collection has: {col.count()} docs")
    inf.add(
        ids=['1'],
        documents=['first document']
    )

    print(f"Collection has: {col.count()} docs")