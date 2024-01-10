import os
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from langchain.document_loaders import DirectoryLoader

from model import llm_model

# Determine the current directory of the script
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
# Set the paths for reader and templates using the current directory
data_dir = os.path.join(src_dir, "data")  # Replace "data" with the relative path to your data directory


#Load the documents.
loader=DirectoryLoader(data_dir,show_progress=True)
documents=loader.load()


#Load the model
chain = llm_model.my_llm_model(documents)

app = FastAPI()

def result_with_sources(response):

  source_list=[]
  for source_item in response['source_documents']:
    source=source_item.metadata['source']

    file_name=os.path.basename(source)

    if file_name not in source_list:
      source_list.append(file_name)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/query')
async def get_query(query_str : str):
    responses = dict(chain(query_str))
    source_list = []
    for source_item in responses['source_documents']:

        source = source_item.metadata['source']
        file_name = os.path.basename(source)
        if file_name not in source_list:
            source_list.append(file_name)

    print("Source list ", source_list)
    print(responses.keys())
    output = {'response': responses.get('result'), 'Source documents': source_list}
    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)