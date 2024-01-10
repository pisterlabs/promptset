from pathlib import Path
from typing import Dict
import modal
from modal import Image, Secret, Stub

image = Image.debian_slim().pip_install(
    # scraping pkgs
    "beautifulsoup4~=4.11.1",
    "httpx~=0.23.3",
    "lxml~=4.9.2",
    # langchain pkgs
    "faiss-cpu~=1.7.3",
    "langchain~=0.0.138",
    "openai~=0.27.4",
    "tiktoken==0.3.0",
    "openai",
    "PyPDF2"
    # TODO Need to add OpenAI's API directly if it gives better results
).run_commands(
    "apt-get update",
    'mkdir /root/DS/',
    'mkdir /DS/',
    'apt-get install -y wget',
    'apt-get install -y curl',
    'wget -o /root/DS/c.pdf https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf',
    'wget -o /DS/c.pdf https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf',
)
stub = Stub(
    name="projectx-langchain",
    image=image,
    secrets=[Secret.from_name("my-openai-secret")],
)
docsearch = None


def scrape_state_of_the_union(textBig) -> str:
    # TODO Need to make it so that the formatting is optimal (remove symbols, etc; whatever needed)
    return textBig


def retrieve_sources(sources_refs: str, texts: list[str]) -> list[str]:
    """
    Map back from the references given by the LLM's output to the original text parts.
    """
    clean_indices = [
        r.replace("-pl", "").strip() for r in sources_refs.split(",")
    ]
    numeric_indices = (int(r) if r.isnumeric()
                       else None for r in clean_indices)
    return [
        texts[i] if i is not None else "INVALID SOURCE" for i in numeric_indices
    ]


def qanda_langchain(query: str, textBig: str, is_PDF: int) -> tuple[str, list[str]]:
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.llms import OpenAI
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores.faiss import FAISS
    import PyPDF2
    import openai
    import os

    import subprocess
    wd = os.getcwd()
    os.chdir("/DS/")
    # subprocess.Popen("ls")
    subprocess.Popen("ls")
    os.chdir(wd)

    inputSource = "/DS/c.pdf"

    # Support caching speech text on disk.
    # ! 0 = NOT A PDF
    if int(is_PDF) == 0:
        speech_file_path = Path("state-of-the-union.txt")

        # Checks if the file exists, if not, calls the function again
        # if speech_file_path.exists():
        #     state_of_the_union = speech_file_path.read_text()
        # else:
        #     print("Writing the file again")
        state_of_the_union = scrape_state_of_the_union(textBig)
        print(state_of_the_union)
        speech_file_path.write_text(state_of_the_union)
    # ! 1 IS A PDF
    elif int(is_PDF) == 1:
        # pdf_handler = PDFHandler()
        # textBig = pdf_handler.process(inputSource)
        # state_of_the_union = textBig
        try:
            reader = PyPDF2.PdfReader(inputSource)
            for page in reader.pages:
                text = page.extract_text()
                state_of_the_union += text
        except Exception as e:
            print("HOLY MOLY {}".format(e))
        print(state_of_the_union)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=24)
    # OG Value: 1000, 0: Quality:Decent
    # tries: 1000,24-same as 1000:0

    # TODO Mess with the chunk_size and chunk_overlap to see if you get better results
    # TODO chunk_size <= 1000, try using 1000+24; chunk_overlap <= 0, try using 1000+<24>, 24 being chunk_overlap
    # TODO cS + cO

    # ! However, it’s important to note that increasing the chunk_overlap parameter will also increase the number of chunks produced from a given input text. This can have an impact on the final result, as it may affect the accuracy of any downstream analysis or processing that you perform on the text data.
    # ! Play around with the sizes of chunk and see if anything comes of it

    print("HAIYAAAA")
    print(state_of_the_union)
    texts = text_splitter.split_text(state_of_the_union)

    global docsearch
    # ? What is docSearch and why is a constant ? I Still dont get what it does

    if not docsearch:
        print("generating docsearch indexer")
        docsearch = FAISS.from_texts(
            texts,
            OpenAIEmbeddings(chunk_size=8),
            # TODO Mess with chunk_size here, Bing says 1000 is good, but I doubt that, need to mess around
            metadatas=[{"source": i} for i in range(len(texts))],
        )

    print("selecting text parts by similarity to query")
    docs = docsearch.similarity_search(query)

    # ! Making a chain here from LC, defining object type, and loading the temperature of the chain
    # ? I think the LLM gets called inside this chain
    chain = load_qa_with_sources_chain(
        # TODO Tinker with the temperature to see if you get better results, temp = randomness
        # ! What is this chain_type = stuff ? Need to check langChain documentation for this
        OpenAI(
            temperature=0.8,
            # streaming=False,
            # maxRetries=3,
            # maxTokens=150
        ),
        # E
        chain_type="stuff"
        # TODO See if changing this actually affects output
        # , max_output_length=1000
    )
    print("running query against Q&A chain.\n")
    # ? The chain is being called here, look ar the results
    result = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    output: str = result["output_text"]
    parts = output.split("SOURCES: ")
    if len(parts) == 2:
        answer, sources_refs = parts
        sources = retrieve_sources(sources_refs, texts)
    elif len(parts) == 1:
        answer = parts[0]
        sources = []
    else:
        raise RuntimeError(
            # TODO find a better way to handle this, without passing it to the LLM
            f"Expected to receive an answer with a single 'SOURCES' block, got:\n{output}"
        )
    return answer.strip(), sources
    # TODO Need to find where the bot says "I DON'T KNOW. SOURCES:", need to add a internal looper there, max 5


@stub.function(gpu="T4", container_idle_timeout=600, secret=modal.Secret.from_name("my-openai-secret"))
@modal.web_endpoint(method="POST")
def cli(varD: Dict):

    # ! Ask to Summary
    if int(varD["mode"]) == 0:
        answer, sources = qanda_langchain(
            varD["query"], varD["text"], varD["mode"]
        )

    # ! Ask to consitution
    elif int(varD["mode"]) == 1:
        answer, sources = qanda_langchain(
            varD["query"], "", varD["mode"]
        )

    answerBack = {"Answer: ": answer}
    edgeCase = ["I don't know", "I DON'T KNOW"]
    for i in edgeCase:
        if i in answer:
            print("Looking again")
            answer, sources = qanda_langchain(
                varD["query"], varD["text"], varD["mode"])
    return answerBack
