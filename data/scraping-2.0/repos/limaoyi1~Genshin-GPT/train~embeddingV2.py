# import json
# from pathlib import Path
#
# import torch.backends
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.schema import Document
# from langchain.vectorstores import Chroma
#
# # 使用 https://github.com/JovenChu/embedding_model_test 的经验
#
# embedding_model_dict = {
#     "MiniLM": "sentence-transformers/all-MiniLM-L6-v2"
# }
#
# EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#
# # 加载JSON文件并解析为Python对象
# with open(file='./../resource/output_en.json', mode='r', encoding="utf-8") as file:
#     json_data = json.load(file)
#
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict['MiniLM'],
#                                    model_kwargs={'device': EMBEDDING_DEVICE})
#
# # file_path = './../resource/output_chs.json'
# # data = json.loads(Path(file_path).read_text(encoding="utf-8"))
#
# # jq 包在windows 的不支持
# # loader = JSONLoader(file_path='./../resource/output_chs.json',jq_schema='.text')
# # data = loader.load()
# # 制作嵌入式向量
# docs = []
# for each in json_data:
#     if each.get('text', '') == '':
#         continue
#     doc = Document(page_content=each.get('npcName', '') + ":\"" + each.get('text', '')+"\"",
#                    metadata={"language": each.get('language', ''), "npcName": each.get('npcName', ''),
#                              "type": each.get('type', '')})
#     docs.append(doc)
#
# import time
# print("start")
#
# # pprint(data)
# start_time = time.time()
# print(start_time)
#
# # pprint(data)
# vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./../resource/dict/v5")
#
# print("exit")
# end_time = time.time()
# print(end_time)
# execution_time = end_time - start_time
# print("Execution time: {:.2f} seconds".format(execution_time))
#
#
