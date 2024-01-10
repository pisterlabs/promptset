from langchain.document_loaders import UnstructuredAPIFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

def summarize(file_path) -> str:
    # 导入文本
    loader = UnstructuredAPIFileLoader(file_path)
    # 文本 转 Document 对象
    document = loader.load()
    
    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 500,
        chunk_overlap=0
    )
    
    # 切分文本
    split_documents = text_splitter.split_documents(document)
    print(f'documents: {len(split_documents)}')
    
    # 加载 llm 模型
    llm = OpenAI(model_name="text-davinci-003", max_tokens=1500)
    
    # 创建总结链
    chain = load_summarize_chain(llm, chain_type="refine", verbose=True)
    
    # 执行总结链
    return chain.run(split_documents[:3])