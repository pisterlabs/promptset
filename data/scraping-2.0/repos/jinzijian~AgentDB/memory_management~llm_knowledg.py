from llms.gpt import call_gpt4
import pinecone
import configparser
import langchain
from langchain.embeddings import OpenAIEmbeddings
from vectorDB.init_emb import get_embeddings_model

def get_action(config, old_task, new_task):
    prompt = '''In facing the following situation, to maintain the practicality of the knowledge base, which action should be taken: update, both save, or delete_old? 
                Directly provide the answer without saying anything else, respond as requested.
                Old task: {}.
                New task: {}.'''
    prompt = prompt.format(old_task, new_task)
    resp = call_gpt4(config, prompt)
    return resp
    
def update():
    pass

def delete():
    pass

def both_save():
    pass

def save_task(config, task, db, index):
    operations = {
    "update": update,
    "delete": delete,
    "both_save": both_save
}
    index = db.Index(index)
    emb_model = get_embeddings_model(config)
    query_emb = emb_model.embed_query(task)
    old_tasks = index.query(
                vector=query_emb,
                top_k=3,
                include_values=True
                )
    for old_task in old_tasks:
        action = get_action(config, old_task, new_task=task)
        result = operations.get(action, lambda: "无效操作")()
    return result
        

