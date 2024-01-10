import weaviate
#from schema import schema
import aux
import requests
import time

def is_valid_weaviate(url):
    try:
        response = requests.get(f"{url}/v1/meta")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
    

def check_batch_result(results: dict):
    if results is not None:
        for result in results:
            if "result" in result and "errors" in result["result"]:
                if "error" in result["result"]["errors"]:
                    pass
                    #print(result["result"])

class weav_db:
    def __init__(self, weaviate_url, credentials):
        connection_retries = 10
        while not is_valid_weaviate(weaviate_url) and connection_retries > 0:
            print("Waiting for Weaviate to come online...")
            connection_retries -= 1
            time.sleep(1)

        self.client = weaviate.Client(
            url = weaviate_url,
            #auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_key),
            additional_headers = {
                "X-OpenAI-Api-Key": credentials['open_ai_api_key']
            },
            startup_period=10
        )
        self.weaviate_url = weaviate_url
        self.open_ai_api_key = credentials['open_ai_api_key']

    #def print_version(self):
    #    print(self.client.meta.get())


    def link_progress_bar(self, progressBar):
        self.progressBar = progressBar

    def class_delete(self, class_name):
        try:
            self.client.schema.delete_class(class_name)
        except:
            print("Class does not exist")

    def class_create(self):
        try:
            self.client.schema.create(schema)
        except:
            print("Class already exists")

    def get_schema(self):
        sch = self.client.schema.get()
        return sch
    
    def mod_schema_properties(self, class_name, schema_config):
        self.client.schema.update_config(class_name, schema_config)
    
    def clear_class(self, class_name):
        all_items, all_additional = self.objects_get(class_name, ["thread_id"], '', ['id'], 0)

        for additional in all_additional:
            self.client.data_object.delete(uuid=additional['uuid'], class_name=class_name)
    
    def schema_add_property(self, class_name, property):

        self.client.schema.property.create(class_name, property)


    def schema_add_class(self, class_obj):
        self.client.schema.create_class(class_obj)


    def objects_set_property(self, class_name, uuid, property_name, property_value):
        try:
            self.client.data_object.update(uuid=uuid, 
                                        class_name=class_name,
                                        data_object={
                                            property_name: property_value
                                            })
        except:
            pass


    def objects_get_pages_with_additional(self, where = '', properties = ["email_id", "user_id", "body_proc"], class_name = "Email", total_documents = 10):
        limit = 10000
        all_documents = []
        all_vectors = []
        offset = 0
        
        while offset < total_documents:
            res = (self.client.query.get(class_name, properties)
                                         .with_limit(limit)
                                         .with_additional("vector")
                                         .with_offset(offset)
                                         .with_where(where)
                                         .do())
            offset += limit
            try:
                all_documents.extend([body['body_proc'] for body in res['data']['Get']['Email']])
                all_vectors.extend([body['_additional']['vector'] for body in res['data']['Get']['Email']])
            except:
                print(res)
            print("Retrieved: ", len(all_documents), len(all_vectors), "/", total_documents)    

        return all_documents, all_vectors



    def objects_get_pages(self, class_name, properties, total_documents = 10):
        limit = 10000
        all_documents = []
        offset = 0

        self.progressBar.setRange(0, total_documents)
        self.progressBar.setValue(0)
        
        while offset < total_documents:
            res = (self.client.query.get(class_name, properties)
                                         .with_limit(limit)
                                         .with_offset(offset)
                                         .do())
            offset += limit
            try:
                all_documents.extend([body[properties[0]] for body in res['data']['Get']['Email']])
            except:
                print(res)
            print("Retrieved: ", len(all_documents), "/", total_documents)    
            self.progressBar.setValue(len(all_documents))
            self.progressBar.setFormat(f"Retrieved: {len(all_documents)} / {total_documents}")

        return all_documents




    def objects_get(self, class_name, properties, where = '', additional = '', total_documents = 100):
        if total_documents == 0:
            total_documents = self.objects_get_count(class_name, where)

        batch_size = 10000
        all_documents = []
        all_additional = []
        offset = 0

        self.progressBar.setRange(0, total_documents)
        self.progressBar.setValue(0)

        while offset < total_documents:

            if where == '' and additional == '':
                res = (self.client.query.get(class_name, properties)
                                            .with_limit(batch_size)
                                            .with_offset(offset)
                                            .do())
            elif where != '' and additional == '':
                res = (self.client.query.get(class_name, properties)
                                            .with_limit(batch_size)
                                            .with_offset(offset)
                                            .with_where(where)
                                            .do())
            elif where == '' and additional != '':
                res = (self.client.query.get(class_name, properties)
                                            .with_limit(batch_size)
                                            .with_offset(offset)
                                            .with_additional(additional)
                                            .do())
            elif where != '' and additional != '':
                res = (self.client.query.get(class_name, properties)
                                            .with_limit(batch_size)
                                            .with_offset(offset)
                                            .with_where(where)
                                            .with_additional(additional)
                                            .do())
            else:
                offset -= batch_size

            offset += batch_size

            #print(res)
            try:
                all_documents.extend([body[properties[0]] for body in res['data']['Get'][class_name]])
                if additional != '':
                    print("checking additional")
                    all_additional.extend([body['_additional'][additional] for body in res['data']['Get'][class_name]])
            except Exception as e:
                print("ERROR: ", e)
            self.progressBar.setValue(len(all_documents))

        if additional == '':
            return all_documents
        else:
            return all_documents, all_additional
    



    def weav_get_by_user_id(self, user_id, class_name = "Email", limit = 1):
        import json
        response = self.client.query.get(class_name, ["email_id", "user_id", "body_proc"]).with_limit(limit).with_where({
            "path": "user_id",
            "operator": "Equal",
            "valueText": user_id
        }).do()
        print(json.dumps(response, indent=2))

    def objects_get_count(self, class_name, where = ''):
        if where == '':
            num_items = (
                self.client.query
                .aggregate(class_name)
                .with_meta_count()
                .do()["data"]["Aggregate"][class_name][0]["meta"]["count"])
        else:
            num_items = (
                self.client.query
                .aggregate(class_name)
                .with_meta_count()
                .with_where(where)
                .do()["data"]["Aggregate"][class_name][0]["meta"]["count"])
        return num_items




    def objects_get_count_distinct(self, class_name, field, where = ''):
        items = self.objects_get(class_name, [field], where, additional='', total_documents=0)

        from collections import Counter
        counts = Counter(items)
        
        #for string, count in counts.items():
        #    print(f"{string}: {count}")
        #print(len(counts), len(items))
            
        return counts











    def upload_data(self, class_name, data_objects, data_ids):
        import time 

        with self.client.batch(
            batch_size = 200, 
            num_workers = 10,
            dynamic = True,
            timeout_retries = 5,
            connection_error_retries = 5,
            callback = check_batch_result
        ) as batch:
            for d, data_object in enumerate(data_objects):
                #print("Uploading data object", d, "of", len(data_objects), end="\r")
                aux.print_progress_bar(0, int(100*d/len(data_objects)), message = f"Uploading data object {d} of {len(data_objects)}")
                batch.add_data_object(
                    data_object,
                    class_name = class_name,
                    uuid = data_ids[d]
                    #uuid=generate_uuid5(question_object)
                )
                time.sleep(0.015)
            print()

    












    # - EVALUATOR - 
    def upload_messages(self, user_messages):
        from weaviate.util import generate_uuid5

        data_objects = [{
            'user_id': msg['user_id'], 
            'email_id': msg['email_id'],
            'body_proc': msg['body_proc']
        } for msg in user_messages]
        data_ids = [generate_uuid5(str(msg['_id'])) for msg in user_messages]

        self.upload_data("Email", data_objects, data_ids)


def oai_query_chat(self, prompt, model = "gpt-3.5-turbo-16k", class_name = "Email"):
    if prompt == "":
        prompt = f"""
        What are the main functions performed? \
        On what objects are the products performed? \
        Format your response as a python list of tuples, example: [("Function", "Object"), ("Function", "Object")].
        Limit your response to 10 tuples.
        """

    import os
    os.environ["OPENAI_API_KEY"] = self.open_ai_api_key

    from langchain.embeddings import OpenAIEmbeddings 
    from langchain.vectorstores.weaviate import Weaviate
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    embeddings = OpenAIEmbeddings()
    vectorstore = Weaviate(self.client, "Email", "body_proc", embeddings)

    chat = ChatOpenAI(
        temperature=0, 
        openai_api_key=self.open_ai_api_key,
        model_name = model
    )

    try:
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=chat, 
            chain_type='stuff',
            retriever=vectorstore.as_retriever(),
        )
        res = retrieval_qa.run(prompt)
        return res
    except Exception as e:
        return ""



def oai_query_compare_llm_models(self, prompt, model = "text-davinci-003", class_name = "Email"):
    if prompt == "":
        prompt = f"""
        What are the main functions performed? \
        On what objects are the products performed? \
        Format your response as a python list of tuples, example: [("Function", "Object"), ("Function", "Object")].
        Limit your response to 10 tuples.
        """

    import os
    os.environ["OPENAI_API_KEY"] = self.open_ai_api_key

    from langchain.embeddings import OpenAIEmbeddings 
    from langchain.vectorstores.weaviate import Weaviate
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI

    embeddings = OpenAIEmbeddings()
    vectorstore = Weaviate(self.client, "Email", "body_proc", embeddings)

    chat = OpenAI(
        temperature=0, 
        openai_api_key=self.open_ai_api_key,
        model_name = model
    )

    try:
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=chat, 
            chain_type='stuff',
            retriever=vectorstore.as_retriever(),
        )
        res = retrieval_qa.run(prompt)
        return res
    except Exception as e:
        return ""





def qna_query(client, user_id, open_ai_api_key, class_name = "Email"):
    ask = {
        "question": "What are the main functions performed? On what objects are the products performed?",
        "properties": ["body_proc"]
    }

    prompt = f"""
    What are the main functions performed? \
    On what objects are the products performed? \
    Format your response as a python list of tuples, example: [("Function", "Object"), ("Function", "Object")].
    Limit your response to 10 tuples.
    """

    prompt = f"""
    What are the main functions performed? \
    On what objects are the functions performed? 
    """

    ask = {
        "question": prompt,
        "properties": ["body_proc"]
    }

    #response = (
    #    client.query
    #    .get(class_name, ["body_proc"])
    #    .with_where({
    #        "path": "user_id",
    #        "operator": "Equal",
    #        "valueText": user_id
    #    })
    #    .with_ask(ask)
    #    .do()
    #)

    response = (
        client.query
        .get(class_name, ["body_proc", "_additional {answer {hasAnswer property result startPosition endPosition} }"])
        .with_ask(ask)
        .with_limit(1)
        .do()
    )

    import json
    print(json.dumps(response, indent=2))