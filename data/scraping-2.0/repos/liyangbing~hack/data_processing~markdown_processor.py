from langchain.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain.text_splitter import MarkdownTextSplitter

class MDProcessor:
    """一个用于处理Markdown文档的类"""

    def __init__(self, chunk_size, chunk_overlap_window):
        self.chunk_size = chunk_size
        self.chunk_overlap_window = chunk_overlap_window

    def load_data(self, file_path):
        """从指定文件路径加载Markdown数据"""
        loader = UnstructuredMarkdownLoader(file_path)
        data = loader.load()
        return data

    def load_data_from_directory(self, directory_path):
        """从指定目录路径加载Markdown数据"""
        loader = DirectoryLoader(directory_path)
        data = loader.load()
        return data

    def split_data(self, document_list):
        """将输入的Markdown文档列表分割成更小的文本块"""
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap_window)
        splitted_data = markdown_splitter.split_documents(document_list)
        return splitted_data

    def display_info(self, data):
        """打印输入数据的信息"""
        print(f'There are {len(data)} documents in your data. ')
        print(
            f'There are {len(data[0].page_content)} characters in your first document. ')

if __name__ == "__main__":
    processor = MDProcessor(chunk_size=100, chunk_overlap_window=10)
    data = processor.load_data('../dataset/data.md')
    splitted_data = processor.split_data(data)
    processor.display_info(splitted_data)

    # 打印第一个文档的相关信息
    print(splitted_data[0])
    print(type(splitted_data[0]))
    print(splitted_data[0].page_content)
    print(splitted_data[0].metadata)
    print(splitted_data[0].metadata["source"])
