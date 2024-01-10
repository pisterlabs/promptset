from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter, SpacyTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader

baseDir = "./files/"

# 加载文档后，您通常会想要对其进行转换以更好地适合您的应用程序。
# 最简单的例子是，您可能希望将长文档分割成更小的块，以适合模型的上下文窗口。
# LangChain 有许多内置的文档转换器，可以轻松地拆分、组合、过滤和以其他方式操作文档。

loader = UnstructuredPDFLoader(baseDir + "index.pdf")
pages = loader.load()
content = pages[0].page_content

# 使用RecursiveCharacterTextSplitter
def RecursiveCharacterTextSplitterDemo():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap  = 20,
        length_function = len,
        add_start_index = True
    )

    texts = text_splitter.create_documents([content])
    print(texts[0])
    print(len(texts))
 
   
# 使用CharacterTextSplitter
def CharacterTextSplitterDemo():
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 100,
        chunk_overlap = 0
    )
    texts = text_splitter.split_text(content)
    print(texts)
    print(len(texts))


# 使用TokenTextSplitter
def TokenTextSplitterDemo():
    text_splitter = TokenTextSplitter.from_tokenizer(
        chunk_size = 100,
        chunk_overlap = 0
    )
    texts = text_splitter.split_text(content)
    print(texts)
    print(len(texts))


# 使用SpacyTextSplitter
# 按字符数 (character count) 分割文本
def SpacyTextSplitterDemo():
    text_splitter = SpacyTextSplitter.from_spacy(
        chunk_size = 100,
        chunk_overlap = 0
    )
    texts = text_splitter.split_text(content)
    print(texts)
    print(len(texts))


# 使用SentenceTransformersTokenTextSplitter
# 按句子数 (sentence count) 分割文本
def SentenceTransformersTokenTextSplitterDemo():
    text_splitter = SentenceTransformersTokenTextSplitter.from_sentence_transformers(
        chunk_overlap = 0
    )
    texts = text_splitter.split_text(content)
    print(texts)
    print(len(texts))


# 使用NLTKTextSplitter
# NLTK我们可以使用基于NLTK 分词器的分割，而不是仅仅根据“\n\n”进行分割。
def NLTKTextSplitterDemo():
    text_splitter = NLTKTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 0
    )
    texts = text_splitter.split_text(content)
    print(texts)
    print(len(texts))




if __name__ == "__main__":
    # RecursiveCharacterTextSplitterDemo()
    # CharacterTextSplitterDemo()
    # TokenTextSplitterDemo()
    # SpacyTextSplitterDemo()
    # SentenceTransformersTokenTextSplitterDemo()
    NLTKTextSplitterDemo()