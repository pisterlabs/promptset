from  llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
from  llama_index.readers.file.base import DEFAULT_FILE_READER_CLS, ImageReader
from llama_index.response.notebook_utils import  display_response, display_image
from llama_index.indices.query.query_transform.base import ImageOutputQueryTransform
import openai

openai.api_key = "x"

image_parser = ImageReader(keep_image=True, parse_text=True)
out_image = image_parser.load_data('/content/drive/MyDrive/colab/aigcData/receipts/111.jpeg')
print(out_image)
file_extractor = DEFAULT_FILE_READER_CLS
file_extractor.update(
    {
        ".jpg": image_parser,
        ".png": image_parser,
        ".jpeg": image_parser,
    }
)

# NOTE: we add filename as metadata for all documents
filename_fn = lambda filename: {'file_name': filename}

receipt_reader = SimpleDirectoryReader(
    input_dir='/content/drive/MyDrive/colab/aigcData/receipts/',
    file_extractor=file_extractor,
    file_metadata=filename_fn,

)
receipt_documents = receipt_reader.load_data()

receipt_index = GPTVectorStoreIndex.from_documents(receipt_documents)
query_engine = receipt_index.as_query_engine()

receipt_response = query_engine.query(
    'When was the last time I went to McDonald\'s and how much did I spend. \
    Also show me the receipt from my visit.'
)

print(receipt_response)