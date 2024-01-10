from langchain.docstore.document import Document

from app.handlers.products import Product
from app.indexer import IDocumentIndexer
from app.store import IStorage


class ProductsIndexer(IDocumentIndexer):
    def __init__(self, storage: IStorage):
        self.storage = storage

    def index(self, catalog_id: str, product: Product):
        results = self._search_products_by_retailer_id(
            catalog_id, [product.product_retailer_id]
        )
        ids = []
        if len(results) > 0:
            ids = [item["_id"] for item in results]
            self.storage.delete(ids=ids)
        doc = Document(page_content=product.title, metadata=product)
        return self.storage.save(doc)

    def index_batch(self, catalog_id: str, products: list[Product]):
        retailer_ids = [product.product_retailer_id for product in products]
        results = self._search_products_by_retailer_id(catalog_id, retailer_ids)
        ids = []
        if len(results) > 0:
            ids = [item["_id"] for item in results]
            self.storage.delete(ids=ids)
        docs = [
            Document(page_content=product.title, metadata=product)
            for product in products
        ]
        return self.storage.save_batch(docs)

    def search(self, search, filter=None, threshold=0.1) -> list[Product]:
        matched_documents = self.storage.search(search, filter, threshold)
        products = [Product.from_metadata(doc.metadata) for doc in matched_documents]
        return products

    def _search_products_by_retailer_id(self, catalog_id, ids):
        search_filter = {
            "metadata.catalog_id": catalog_id,
            "metadata.product_retailer_id": ids,
        }
        return self.storage.query_search(search_filter)

    def delete(self, catalog_id, product_retailer_id):
        results = self._search_products_by_retailer_id(
            catalog_id, [product_retailer_id]
        )
        ids = []
        if len(results) > 0:
            ids = [item["_id"] for item in results]
            self.storage.delete(ids=ids)
        return ids

    def delete_batch(self, catalog_id, product_retailer_ids):
        results = self._search_products_by_retailer_id(catalog_id, product_retailer_ids)
        ids = []
        if len(results) > 0:
            ids = [item["_id"] for item in results]
            self.storage.delete(ids=ids)
        return ids
