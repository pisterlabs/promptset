from langchain.document_loaders import UnstructuredFileLoader,UnstructuredWordDocumentLoader,Docx2txtLoader,PyPDFLoader,PyMuPDFLoader,AmazonTextractPDFLoader,MathpixPDFLoader
import mimetypes
WORD_MIMETYPES=['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.template',
                'application/vnd.ms-word.document.macroenabled.12',
                'application/vnd.ms-word.template.macroenabled.12'
                ]
class DocumentLoaders:
    def __init__(self,file_path) -> None:
        file_type, _=mimetypes.guess_type(file_path)           
        if file_type=='application/msword':
            loader = UnstructuredWordDocumentLoader(file_path=file_path,mode="paged")
        elif file_type.lower() in WORD_MIMETYPES:
            loader=Docx2txtLoader(file_path=file_path)
        elif file_type.lower()=='application/pdf':
            # loader=PyPDFLoader(file_path=file_path)
            # loader=PyMuPDFLoader(file_path=file_path)
            # loader=AmazonTextractPDFLoader(file_path=file_path)
            loader=MathpixPDFLoader(file_path=file_path)
        else:
            loader = UnstructuredFileLoader(file_path=file_path,mode="paged")
        self.loader=loader
        pass
    def getDocumentsLoader(self):        
        return self.loader










