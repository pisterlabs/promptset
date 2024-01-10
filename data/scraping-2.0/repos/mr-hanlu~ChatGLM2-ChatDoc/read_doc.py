from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import uuid
from text2vec.myembedding import MyEmbeddings
import chromadb


class document2data:
    """
    文档读取和切分的类
    """

    def pages2data(self, pages, chunk_size=1000, chunk_overlap=150):
        """
        读取一个解析出来的文件
        :param pages: 读取一个解析出来的文件
        :param chunk_size: 每个字符串块的长度
        :param chunk_overlap: 字符串之间的重叠长度
        :return: ids,documents,metadatas 列表
        """

        r_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_pages = r_splitter.split_documents(pages)
        # print(split_pages)
        # print(len(split_pages))

        ids = []
        documents = []
        metadatas = []
        docstr = ""
        for page in split_pages:
            id = str(uuid.uuid1())
            document = page.page_content
            docstr += document
            metadata = page.metadata

            ids.append(id)
            documents.append(document)
            metadatas.append(metadata)
        return ids, documents, metadatas, docstr

    def read_pdf(self, pdf_path, chunk_size=1000, chunk_overlap=150):
        """
        读取一个pdf文件，返回一个分割后的字符串列表
        :param pdf_path: pdf文件路径
        :param chunk_size: 每个字符串块的长度
        :param chunk_overlap: 字符串之间的重叠长度
        """
        from langchain.document_loaders import PyPDFLoader
        loder = PyPDFLoader(pdf_path)
        pages = loder.load()
        return self.pages2data(pages, chunk_size, chunk_overlap)

    def read_md(self, markdown_path, chunk_size=1000, chunk_overlap=150):
        """
        读取一个markdown文件，返回一个分割后的字符串列表
        :param markdown_path: markdown文件路径
        :param chunk_size: 每个字符串块的长度
        :param chunk_overlap: 字符串之间的重叠长度
        """
        from langchain.document_loaders import UnstructuredMarkdownLoader
        loder = UnstructuredMarkdownLoader(markdown_path)
        pages = loder.load()
        return self.pages2data(pages, chunk_size, chunk_overlap)

    def read_csv(self, csv_path, chunk_size=1000, chunk_overlap=150):
        """
        读取一个csv文件，返回一个分割后的字符串列表
        :param csv_path: csv文件路径
        :param chunk_size: 每个字符串块的长度
        :param chunk_overlap: 字符串之间的重叠长度
        """
        from langchain.document_loaders import CSVLoader
        loder = CSVLoader(csv_path)
        pages = loder.load()
        return self.pages2data(pages, chunk_size, chunk_overlap)

    def read_docx(self, docx_path, chunk_size=1000, chunk_overlap=150):
        """
        读取一个docx文件，返回一个分割后的字符串列表
        :param docx_path: docx文件路径
        :param chunk_size: 每个字符串块的长度
        :param chunk_overlap: 字符串之间的重叠长度
        """
        from langchain.document_loaders import UnstructuredWordDocumentLoader
        loder = UnstructuredWordDocumentLoader(docx_path)
        pages = loder.load()
        return self.pages2data(pages, chunk_size, chunk_overlap)

    def read_txt(self, txt_path, chunk_size=1000, chunk_overlap=150):
        """
        读取一个txt文件，返回一个分割后的字符串列表
        :param txt_path: txt文件路径
        :param chunk_size: 每个字符串块的长度
        :param chunk_overlap: 字符串之间的重叠长度
        """
        from langchain.document_loaders import TextLoader
        loder = TextLoader(txt_path)
        pages = loder.load()
        return self.pages2data(pages, chunk_size, chunk_overlap)

    def read_other(self, other_path, chunk_size=1000, chunk_overlap=150):
        """
        选择上传文件，会自动选择加载器，返回一个分割后的字符串列表
        :param other_path: 文件路径
        :param chunk_size: 每个字符串块的长度
        :param chunk_overlap: 字符串之间的重叠长度
        """
        from langchain.document_loaders import UnstructuredFileLoader
        loder = UnstructuredFileLoader(other_path)
        pages = loder.load()
        return self.pages2data(pages, chunk_size, chunk_overlap)

    def read_url(self, url, chunk_size=1000, chunk_overlap=150):
        """
        读取一个pdf文件，返回一个分割后的字符串列表
        :param url: web url
        :param chunk_size: 每个字符串块的长度
        :param chunk_overlap: 字符串之间的重叠长度
        """
        from langchain.document_loaders import WebBaseLoader
        loder = WebBaseLoader(url)
        pages = loder.load()
        return self.pages2data(pages, chunk_size, chunk_overlap)


class chorma_loader(document2data):
    """
    文本编码特征存数据库
    """

    def __init__(self, client_path):
        """
        初始化
        :param client_path: 客户端保存路径
        """
        super().__init__()
        # 加载embdding模型
        self.model = MyEmbeddings("./text2vec/shibing624/text2vec-base-chinese")
        # 加载客户端，没有就创建
        self.chroma_client = chromadb.PersistentClient(path=client_path)
        # 获取客户端数据库列表
        print(self.chroma_client.list_collections())

    def load_or_create_collection(self, collection_name, ids=None, documents=None, metadatas=None):
        """
        加载或创建一个collection
        :param collection_name: 数据库名称
        下面三个参数默认为None,None表示不添加数据，只加载或创建数据库
        :param ids: ids列表
        :param documents: 文本列表
        :param metadatas: 元数据列表
        :return: 返回collection数据库
        """

        # 加载或创建collection
        coll_list = [c.name for c in self.chroma_client.list_collections()]
        if collection_name in coll_list:
            collection = self.chroma_client.get_collection(name=collection_name,
                                                           embedding_function=self.model.embed_documents)
        else:
            collection = self.chroma_client.create_collection(name=collection_name,
                                                              metadata={"hnsw:space": "cosine"},
                                                              embedding_function=self.model.embed_documents)

            # 添加数据
            collection.add(
                # 切分后的文档
                documents=documents,
                # 一些关于数据的基本信息/附加信息
                metadatas=metadatas,
                # 索引
                ids=ids
            )

        # # 获取数据库内容
        # doc = collection.get()
        # # 数据长度
        # print(doc["ids"])

        # print(collection.peek())
        print(collection.count())

        # 传给langchain的chroma,langchain的chroma只能有一个数据库
        langchain_chroma = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.model,
        )

        return langchain_chroma


if __name__ == '__main__':
    # pdf_path = r'../High-Resolution.pdf'
    pdf_path = r'C:\Users\user\AppData\Local\Temp\gradio\0b4a044b3bbfd3896659ceb3d4a8f4f91c37c018\High-Resolution.pdf'
    # url = r"https://docs.trychroma.com/usage-guide#creating-inspecting-and-deleting-collections"
    chunk_size = 1000
    chunk_overlap = 150
    k = 3
    client_path = r"./chroma"
    # 创建或加载客户端
    chormadb = chorma_loader(client_path)

    # 读取文档返回id，文本，和元数据(数据的一些信息)
    ids, documents, metadatas, docstr = chormadb.read_pdf(pdf_path, chunk_size, chunk_overlap)
    # ids = None
    # 加载或创建数据库
    collection = chormadb.load_or_create_collection("test", ids, documents, metadatas)

    while True:
        query_texts = ["苹果", "橘子"]

        # 使用langchain的chroma，可以使用最大边际相关性查询
        docs = collection.max_marginal_relevance_search(query_texts[0], k=k, fetch_k=5)

        # 使用chroma
        res = collection.query(
            query_texts=query_texts,
            # 输出的结果数
            n_results=k,
            # # 根据文本内容筛选
            # where_document={"$contains": "水果"},
            # # 按照metadata筛选
            # where={"page": 1},
        )

        print(len(res["ids"][0]))
        print(res["distances"][0])
        # ['ids', 'distances', 'metadatas', 'embeddings', 'documents']
        print(res.keys())
        exit()

    # ids, documents, metadatas = read_url(url, chunk_size, chunk_overlap)
    # print(len(ids))
    # print(documents)
    # print(metadatas)
