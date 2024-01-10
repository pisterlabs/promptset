import os
import re
import sys
import base64
import shutil
os.environ['OPENAI_API_KEY'] = {YOUR_OPENAI_KEY}
from langchain.llms import OpenAI
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, ServiceContext
from llama_index.readers.file.base import (
    DEFAULT_FILE_EXTRACTOR, 
    ImageParser,
)
from llama_index.indices.query.query_transform.base import (
    ImageOutputQueryTransform,
)
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
image_parser = ImageParser(keep_image=False, parse_text=True)
file_extractor = DEFAULT_FILE_EXTRACTOR
file_extractor.update(
{
    ".jpg": image_parser,
    ".png": image_parser,
    ".jpeg": image_parser,
})
source_folder = 'data/input_images'
indexed_folder = 'data/indexed'
index_filepath = './image_index.json'
# add filename as metadata for all documents
filename_fn = lambda filename: {'file_name': filename}
image_reader = SimpleDirectoryReader(
    input_dir=source_folder,
    file_extractor=file_extractor, 
    file_metadata=filename_fn,
    required_exts=['.jpg', '.png', '.jpeg']
)
def move_indexed_files():
    # fetch all files
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".jpeg"):
            # construct full file path
            source = source_folder + '/' + file_name
            destination = indexed_folder + '/' + file_name
            # move only files
            if os.path.isfile(source):
                shutil.move(source, destination)
                #print('Moved:', file_name)

# only load from index flag
image_index = None
if len(sys.argv) >= 2:
    if sys.argv[1].lower() == '-l' or sys.argv[1].lower() == '-load':
        image_index = GPTSimpleVectorIndex.load_from_disk(index_filepath)
# load from disk and add any new images
if (image_index == None):
    image_documents = image_reader.load_data()
    if os.path.isfile(index_filepath):
        image_index = GPTSimpleVectorIndex.load_from_disk(index_filepath)
        for document in image_documents:
            image_index.insert(document) 
    else:
        image_index = GPTSimpleVectorIndex.from_documents(image_documents, service_context=service_context)
    image_index.save_to_disk(index_filepath)
    move_indexed_files()

while True:
    query = input('Enter your query: ')
    if query == "quit" or query == "exit":
        break
    #response = image_index.query(
    #    'Is there a map of NUS shuttle bus routes?',
    #    query_transform=ImageOutputQueryTransform(width=600)
    #)
    response = image_index.query(
        query,
        query_transform=ImageOutputQueryTransform(width=600),
        similarity_cutoff=0.75
    )
    x = None
    if response.response:
        x = re.search('<img src="(.+)".+"\d+"', response.response)
    if x:
        #print(x.group(1))
        image_path = x.group(1).replace("\\","/").replace(source_folder, indexed_folder)
        print(image_path)
        with open(image_path, "rb") as image_file:
            data = base64.b64encode(image_file.read())
            #print(data.decode("utf-8"))
            with open('output.txt','w') as f:
                f.write(data.decode("utf-8"))
            print('Base64 encoded image written to output.txt')
    else:
        print('No matching image found.')
