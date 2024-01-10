
from ragas.metrics.critique import AspectCritique
from llama_index import load_index_from_storage
from llama_index import  StorageContext, set_global_service_context, ServiceContext
from datasets import Dataset
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
import json
from tqdm import tqdm
import os

def write_to_json(filename,finetuning_dataset):
    
    database = json.load(open(filename))
    database.extend(finetuning_dataset)
    with open(filename,'w') as file:
        json.dump(database, file, indent=4)

if __name__ == "__main__":
    
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    service_context = ServiceContext.from_defaults(embed_model=embed_model,)
    critic = AspectCritique(name="filter", definition="Does the submission contain information that can be derived from input?")
    dataset = Dataset.from_json("wikidata/indices/subset.json")
    filename = "finetuning_dataset.json"
    
    if not os.path.exists(filename):
        with open(filename,"w") as file:
            json.dump([], file)
        
    
    
    batch_size=100
    max_ragas_score = 0.8
    threshold=0.8
    for batch in tqdm(range(0,len(dataset)+1, batch_size)):
        datapath=f"./sample-{batch}.index/"
        # create storage context using default stores
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=datapath),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=datapath),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=datapath),
        )
        set_global_service_context(service_context)
        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever(similarity_top_k=1)
        subsample = dataset.select(range(batch, min(len(dataset), batch+batch_size)))
        finetuning_dataset = []

        try:
            for item in subsample:
                if item["ragas_score"] <= max_ragas_score:
        
                    node = retriever.retrieve(item["Answer"])[0]
                    filter = critic.score_single({"question":node.get_content(),"answer":item["Answer"]})

                    # if node.get_score()>=threshold:
                    if filter:
                        pos_chunk = node.to_dict()
                    else:
                        continue

                    
        
                    retrieved_chunks = item["chunks"]
                    # hard negatives : till positive hash
                    hard = True
                    hard_negatives,negatives = [], []
                    for node in retrieved_chunks:
        
                        if node["node"]["hash"] == pos_chunk["node"]["hash"]:
                            hard = False
                            continue
        
                        if hard:
                            hard_negatives.append(node)
                        else:
                            negatives.append(node)
        
                    sample = {"Question":item["Question"], "Answer":item["Answer"],
                            "Context":item["Context"],
                            "Conversation_no":item["Conversation_no"],
                            "Turn_no":item["Turn_no"],
                            "Positives":[pos_chunk["node"]["text"]],
                            "Negatives":[chunk["node"]["text"] for chunk in negatives],
                            "Hard_negatives":[chunk["node"]["text"] for chunk in hard_negatives]}
                    finetuning_dataset.append(sample)
                    
            write_to_json(filename, finetuning_dataset)
        except Exception as e:
            print(e)
                
                
    