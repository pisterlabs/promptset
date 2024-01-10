import unittest
from unittest.mock import Mock
from langchain.docstore.document import Document
from app.store import IStorage
from app.indexer.products import ProductsIndexer
from app.handlers.products import Product


class TestProductsIndexer(unittest.TestCase):
    def setUp(self):
        self.mock_storage = Mock(spec=IStorage)
        self.indexer = ProductsIndexer(self.mock_storage)

    def test_index(self):
        catalog_id = "789"
        mock_product = Product(
            facebook_id="123456789",
            title="Test Product",
            org_id="123",
            channel_id="456",
            catalog_id=catalog_id,
            product_retailer_id="999",
        )
        mock_document = Document(page_content=mock_product.title, metadata=mock_product)

        self.mock_storage.save.return_value = mock_document
        self.mock_storage.query_search.return_value = []

        result = self.indexer.index(catalog_id, mock_product)

        self.mock_storage.save.assert_called_once_with(mock_document)
        self.mock_storage.query_search.assert_called_once_with(
            {
                "metadata.catalog_id": catalog_id,
                "metadata.product_retailer_id": [mock_product.product_retailer_id],
            }
        )
        self.assertEqual(result, mock_document)

    def test_index_updating(self):
        catalog_id = "789"
        mock_document_id = "aa01ca00-fd85-457d-a4a1-47b27d9b6121"
        mock_product = Product(
            facebook_id="123456789",
            title="Test Product",
            org_id="123",
            channel_id="456",
            catalog_id=catalog_id,
            product_retailer_id="999",
        )
        mock_document = Document(page_content=mock_product.title, metadata=mock_product)

        self.mock_storage.save.return_value = mock_document
        self.mock_storage.query_search.return_value = [{"_id": mock_document_id}]

        result = self.indexer.index(catalog_id, mock_product)

        self.mock_storage.save.assert_called_once_with(mock_document)
        self.mock_storage.query_search.assert_called_once_with(
            {
                "metadata.catalog_id": catalog_id,
                "metadata.product_retailer_id": [mock_product.product_retailer_id],
            }
        )
        self.assertEqual(result, mock_document)

    def test_index_batch(self):
        catalog_id = "789"
        mock_products = [
            Product(
                facebook_id="1234567891",
                title="Test Product 1",
                org_id="123",
                channel_id="456",
                catalog_id=catalog_id,
                product_retailer_id="998",
            ),
            Product(
                facebook_id="1234567892",
                title="Test Product 2",
                org_id="123",
                channel_id="456",
                catalog_id=catalog_id,
                product_retailer_id="999",
            ),
        ]
        mock_documents = [
            Document(page_content=product.title, metadata=product)
            for product in mock_products
        ]
        self.mock_storage.save_batch.return_value = mock_documents
        self.mock_storage.query_search.return_value = []

        result = self.indexer.index_batch(catalog_id, mock_products)

        self.mock_storage.save_batch.assert_called_once_with(mock_documents)
        self.assertEqual(result, mock_documents)

    def test_index_batch_updating(self):
        catalog_id = "789"
        mock_document_id = "aa01ca00-fd85-457d-a4a1-47b27d9b6121"
        mock_products = [
            Product(
                facebook_id="1234567891",
                title="Test Product 1",
                org_id="123",
                channel_id="456",
                catalog_id=catalog_id,
                product_retailer_id="998",
            ),
            Product(
                facebook_id="1234567892",
                title="Test Product 2",
                org_id="123",
                channel_id="456",
                catalog_id=catalog_id,
                product_retailer_id="999",
            ),
        ]
        mock_documents = [
            Document(page_content=product.title, metadata=product)
            for product in mock_products
        ]
        self.mock_storage.save_batch.return_value = mock_documents
        self.mock_storage.query_search.return_value = [{"_id": mock_document_id}]

        result = self.indexer.index_batch(catalog_id, mock_products)

        self.mock_storage.save_batch.assert_called_once_with(mock_documents)
        self.assertEqual(result, mock_documents)

    def test_search(self):
        mock_search_query = "Test Query"
        mock_document = Document(
            page_content="Document 1",
            metadata={
                "facebook_id": "123456789",
                "title": "Title 1",
                "org_id": "123",
                "channel_id": "456",
                "catalog_id": "789",
                "product_retailer_id": "999",
            },
        )
        self.mock_storage.search.return_value = [mock_document]

        result = self.indexer.search(mock_search_query)

        self.mock_storage.search.assert_called_once_with(mock_search_query, None, 0.1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].title, mock_document.metadata["title"])

    def test_delete(self):
        catalog_id = "789"
        product_retailer_id = "pd123"
        mock_document_id = "b554c118-b6d7-40a7-8d3e-560d63f91723"
        self.mock_storage.query_search.return_value = [{"_id": mock_document_id}]
        self.mock_storage.delete.return_value = True
        result = self.indexer.delete(catalog_id, product_retailer_id)
        self.assertEqual(result, [mock_document_id])

    def test_delete_batch(self):
        catalog_id = "789"
        product_retailer_ids = ["pd123", "pd124"]
        mock_document_id1 = "b554c118-b6d7-40a7-8d3e-560d63f91723"
        mock_document_id2 = "5c69174c-1dbd-4967-b251-d6b9132366b2"
        self.mock_storage.query_search.return_value = [
            {"_id": mock_document_id1},
            {"_id": mock_document_id2},
        ]
        self.mock_storage.delete.return_value = True
        result = self.indexer.delete_batch(catalog_id, product_retailer_ids)
        self.assertEqual(result, [mock_document_id1, mock_document_id2])
