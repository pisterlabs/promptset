from flask import request
import json
from utils.vectorDB import Repo_handler
from utils.langchain import LangChain

async def create_data():
    """Creates new data from the request body."""
    try:
        # set data
        data = request.data
        data_dict = json.loads(data)

        repo = data_dict['repo_name']
        repo_url = data_dict['repo_url']

        repo_data = {'repo_name': repo, 'repo_url': repo_url}
        repo_handler = Repo_handler(repo_data)
        db = repo_handler.get_repo_embedding(dataset_path = repo_handler.dataset_path)
        
        if db:
            qa_data = {"qa_data": data_dict['qa_data'], "db": db}
            langchain = LangChain(qa_data)
            qa = langchain.retrieval_model()
            result = qa({"query": data_dict['qa_data']})
            print(result)
            answer = result['result']
            print(answer)
        
            docs = set()
            for doc in result['source_documents']:
                docs.add(doc.metadata['source'])

            unique_docs = list(docs)
            print(unique_docs)

            return {'message': 'Data created successfully', 'data': answer, 'docs': unique_docs, 'query': result['query']}


        else:
            return {'message': 'Data created unsuccessfully', 'data': 'No data found'}


    except Exception as e:
        # Handle any exceptions or errors
        return {'error': str(e)}, 400   
