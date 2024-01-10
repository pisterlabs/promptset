
from nameFormat import NameFormat

from FaissDB_Utils import FaissDB_Utils
from keys import OpenAI_API_KEY



faissDB_Utils = FaissDB_Utils(api_key=OpenAI_API_KEY)




persist_directory='liudongxingpipei'
filename=NameFormat.format(name=persist_directory)

#db = faissDB_Utils.path_to_db(directory_path='data/file', userName='clw')

# 查询数据
query = "流动性覆盖率的公式？"
results = faissDB_Utils.search_documents(query=query, userName='gfile')

for i, result in enumerate(results):
    print(f"Result {i + 1}:")
    print(result.page_content)
    print(f"Metadata: {result.metadata}")
    file_path = result.metadata['source']
    file_name = file_path.split('/')[-1]  # 使用分隔符 '/' 分割路径并获取最后一个部分作为文件名

    print(f"File Name: {file_name}")
    print("\n")