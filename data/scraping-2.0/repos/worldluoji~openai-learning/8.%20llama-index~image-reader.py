import openai, os
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex
from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR, ImageParser
from llama_index.response.notebook_utils import display_response, display_image
from llama_index.indices.query.query_transform.base import ImageOutputQueryTransform

openai.api_key = os.environ.get("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 要能够索引图片，我们引入了 ImageParser 这个类，这个类背后，其实是一个基于 OCR 扫描的模型 Donut。它通过一个视觉的 Encoder 和一个文本的 Decoder，这样任何一个图片能够变成一个一段文本，然后我们再通过 OpenAI 的 Embedding 把这段文本变成了一个向量。
image_parser = ImageParser(keep_image=True, parse_text=True)

file_extractor = DEFAULT_FILE_EXTRACTOR
file_extractor.update(
{
    ".jpg": image_parser,
    ".png": image_parser,
    ".jpeg": image_parser,
})


# NOTE: we add filename as metadata for all documents
filename_fn = lambda filename: {'file_name': filename}

receipt_reader = SimpleDirectoryReader(
    input_dir='./pics', 
    file_extractor=file_extractor, 
    file_metadata=filename_fn,
)
receipt_documents = receipt_reader.load_data()


receipts_index = GPTSimpleVectorIndex.from_documents(receipt_documents)
receipts_response = receipts_index.query(
    'When was the last time I went to McDonald\'s and how much did I spend. \
    Also show me the receipt from my visit.',
    query_transform=ImageOutputQueryTransform(width=400)
)

display_response(receipts_response)

# 打印ImageParser解析出的图片内容
print('*' * 13)
output_image = image_parser.parse_file('./pics/100-receipt.jpg')
print(output_image.text)


# https://llamahub.ai/