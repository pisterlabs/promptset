def search_cosine_similarity(query,split_docs,embeddings):  ##query-str,split_docs-list,embeddings-embeddings()
    split_docs_content = [content['content'] for content in split_docs]
    embed_docs = embeddings.embed_documents(split_docs_content)
    embed_query= embeddings.embed_query(query)
    
    from openai.embeddings_utils import cosine_similarity
    
    cos_index = []
    for embed_doc in embed_docs:
        cos_index.append(cosine_similarity(embed_doc,embed_query))
    
    #这边是根据大小建立索引
    idx = sorted(range(len(cos_index)),key=lambda k:cos_index[k]) #根据cos_index的大小进行排序
    final_similar_list = []
    for index in idx[-3:]:
        unit = {}
        unit['sentences']=split_docs_content[index]
        unit['source']=split_docs[index]['source']
        unit['score']=cos_index[index]
        final_similar_list.append(unit)
    
    return final_similar_list