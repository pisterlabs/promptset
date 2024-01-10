import os
import uuid
from io import StringIO
from werkzeug.utils import secure_filename
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from openai_utils import get_embedding

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def process_file(file_storage):
    filename = secure_filename(file_storage.filename)
    temp_file_path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(temp_file_path)

    rsrcmgr = PDFResourceManager()
    laparams = LAParams()

    embeddings = []
    with open(temp_file_path, 'rb') as fp:
        for i, page in enumerate(PDFPage.get_pages(fp)):
            metadata = {}
            print(f"Processing page {i+1}...")
            with StringIO() as output_string:
                with TextConverter(rsrcmgr, output_string, laparams=laparams) as device:
                    interpreter = PDFPageInterpreter(rsrcmgr, device)
                    interpreter.process_page(page)
                
                text = output_string.getvalue()

                id = str(uuid.uuid4())
                metadata["filename"] = filename
                metadata["page"] = i+1

                embeddings.append((id, get_embedding(text), metadata))
    
    os.remove(temp_file_path)

    return embeddings