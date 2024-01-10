import dlt
import os
from typing import Dict, List
from google_drive_connector import download_pdf_from_google_drive, get_pdf_uris
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator


# Add your own folder id (from Google Drive URL) (or use config.toml)
folder_id=None


def safely_query_index(index, query):
    try:
        return index.query(query).strip()
    except Exception:
        return []

def process_one_pdf_to_structured(path_to_pdf:str) -> Dict:
    loader = UnstructuredPDFLoader(path_to_pdf)
    index = VectorstoreIndexCreator().from_loaders([loader])
    return {
        "file_name": path_to_pdf.split("/")[-1],
        "recipient_company_name": safely_query_index(index, "Who is the recipient of the invoice? Just return the name"),
        "invoice_amount": safely_query_index(index, "What is the total amount of the invoice? Just return the amount as decimal number, no currency or text"),
        "invoice_date": safely_query_index(index, "What is the date of the invoice? Just return the date"),
        "invoice_number": safely_query_index(index, "What is the invoice number? Just return the number"),
        "service_description": safely_query_index(index, "What is the description of the service that this invoice is for? Just return the description"),
    }

def process_all_pdfs_to_structured(path_to_pdfs:str)->List[Dict]:
    for file in os.listdir(path_to_pdfs):
        if file.endswith(".pdf"):
            yield process_one_pdf_to_structured(os.path.join(path_to_pdfs,file))
    return []


def download_and_process_one_pdf(file_id, file_name, local_folder_to_store_pdfs:str="./data/invoices", delete_after_extraction=True):
    download_pdf_from_google_drive(file_id, file_name, local_folder_to_store_pdfs)
    structured_data = process_one_pdf_to_structured(os.path.join(local_folder_to_store_pdfs, file_name))
    if delete_after_extraction:
        os.remove(os.path.join(local_folder_to_store_pdfs, file_name))
    yield structured_data

@dlt.source
def invoice_tracking_source(drive_folder_id=dlt.config.value, delete_after_extraction=True):
    return invoice_tracking_resources(drive_folder_id, delete_after_extraction)


@dlt.resource(write_disposition="append")
def invoice_tracking_resources(drive_folder_id, delete_after_extraction):
    uris = get_pdf_uris(drive_folder_id)
    for file_name, file_id in uris.items():
        yield download_and_process_one_pdf(file_id, file_name, delete_after_extraction=delete_after_extraction)


if __name__ == "__main__":
    pipeline = dlt.pipeline(pipeline_name="invoice_tracking", destination="duckdb", dataset_name="invoice_tracking_data")
    # print(list(invoice_tracking_source()))
    load_info = pipeline.run(invoice_tracking_source())
    print(load_info)
