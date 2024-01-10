from langchain.document_loaders import PyPDFLoader, DirectoryLoader


def scan_pdf_from_folder(folder_path):
    try:
        loader = DirectoryLoader(
            f"{folder_path}/", glob="*.pdf", loader_cls=PyPDFLoader
        )
        documents = loader.load()

        if not documents:
            print("No documents found")
            return []

    except Exception as e:
        print(f"Error while scanning folder: {folder_path}\n{e}")
        return []

    print(f"length doc: {len(documents)}")
    return documents


def process_pdf(folder_path, filename):
    try:
        documents = scan_pdf_from_folder(folder_path)
        processed_pdf = []

        for doc in documents:
            processed_pdf.append({"text": doc.page_content, "file": filename})

        # print(f"processed_pdf: {processed_pdf}")
        return processed_pdf

    except Exception as e:
        print(f"Error while processing markdown: {e}")
