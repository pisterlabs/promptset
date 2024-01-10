from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from transformers import pipeline


def main():
    # Step 1: Load documents
    # loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
    loader = DirectoryLoader("example_data", loader_cls=PyPDFLoader)

    # Step 2: Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(loader.load())

    # Step 3: Embed and store splits
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cuda'})

    # Use Instruct Embeddings
    # from langchain.embeddings import HuggingFaceInstructEmbeddings
    #
    # model_name = "hkunlp/instructor-large"
    # model_kwargs = {'device': 'cpu'}
    # encode_kwargs = {'normalize_embeddings': True}
    # hf = HuggingFaceInstructEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs
    # )

    # Store splits
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # from langchain.embeddings import OpenAIEmbeddings
    # from langchain.vectorstores import Chroma
    # vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())


    ##############################################

    # Step 4: Load LLM model from HF
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                 device_map='auto',
                                                 torch_dtype=torch.float16,
                                                 use_auth_token=True,
                                                 load_in_8bit=True,
                                                 # load_in_4bit=True
                                                 )

    # from langchain.chat_models import ChatOpenAI
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Step 5: Create HF pipeline
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=1024,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                    )
    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                        retriever=vectorstore.as_retriever())

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=vectorstore.as_retriever(),
    #     chain_type_kwargs={"prompt": prompt}
    # )

    query = "question?"
    result = chain({"query": query}, return_only_outputs=True)




if __name__ == "__main__":
    main()

# EOF
