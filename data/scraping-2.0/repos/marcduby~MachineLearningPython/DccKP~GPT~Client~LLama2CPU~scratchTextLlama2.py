

# imports
from langchain import PromptTemplate
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS 
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
import time 
import os

# for AWS
ENV_DIR_CODE = os.environ.get('DIR_CODE')
ENV_DIR_PUBMED = os.environ.get('DIR_PUBMED')

# local imports
dir_code = "/home/javaprog/Code/PythonWorkspace/"
if ENV_DIR_CODE:
    dir_code = ENV_DIR_CODE
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/GPT/')
import dcc_gpt_lib
import dcc_langchain_lib

# constants
DIR_DATA = "/home/javaprog/Data/"
DIR_DOCS = DIR_DATA + "ML/Llama2Test/Genetics/Docs"
DIR_VECTOR_STORE = DIR_DATA + "ML/Llama2Test/Genetics/VectorStore"
FILE_MODEL = DIR_DATA + "ML/Llama2Test/Model/llama-2-7b-chat.ggmlv3.q8_0.bin"
FILE_MODEL = DIR_DATA + "ML/Llama2Test/Model/llama-2-13b-chat.ggmlv3.q8_0.bin"
PROMPT = """Use the following piece of information to anser the user's question.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing elase.
Helpful answer:
"""
GPT_PROMPT = """
Below are the abstracts from different research papers on gene {gene}. 
Please read through the abstracts and as a genetics researcher write a 100 word summary that synthesizes the key findings of the papers on the biology of gene {gene}
{abstracts}
"""

GPT_PROMPT = """
Below are the abstracts from different research papers on genetics. 
Please read through the abstracts and as a genetics researcher write a 100 word summary that synthesizes the key biology findings of the papers
{abstracts}
"""

GPT_PROMPT = "{text_prompt}"

# methods
def get_inference_summary_prompt(gene, text_input, max_tokens=512, temperature=0.1,  log=False):
    '''
    do the llm inference
    '''
    if log:
        print("doing llm inference using gene: {} and abstracts: \n{}\n".format(gene, text_input))

    if log:
        dcc_gpt_lib.print_count_words_in_string(text=text_input, name="combined abstracts")

    # get the prompt
    # prompt = dcc_langchain_lib.PROMPT_GENE_TEXT
    prompt = dcc_langchain_lib.PROMPT_SIMPLE_GENE_TEXT
    # prompt = dcc_langchain_lib.PROMPT_SIMPLE_BULLETPOINT_GENE_TEXT
    prompt_template = PromptTemplate(
        input_variables = ['gene', 'text'],
        template = prompt
    )

    # log
    if log:
        print("Using prompt template: \n{}\n".format(prompt_template))

    # get the llm
    llm = dcc_langchain_lib.load_local_llama_model(dcc_langchain_lib.FILE_LLAMA2_7B_CPU, temperature=temperature)
    # llm = dcc_langchain_lib.load_local_llama_model(dcc_langchain_lib.FILE_LLAMA2_13B_CPU, max_new_tokens=max_tokens)

    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    # run the chain
    result = llm_chain.run({'gene': gene, 'text': text_input})

    # return
    return result


# main
if __name__ == "__main__":
    # initialize
    # dir_db = DIR_VECTOR_STORE
    file_model = FILE_MODEL
    prompt = PROMPT
    num_abstracts_per_summary = 2
    max_per_level = 50
    id_run = 7
    num_level = 0

    # test data
    gene = 'PPARG'
    data = "Clinical exome sequencing routinely identifies missense variants in disease-related genes, but functional characterization is rarely undertaken, leading to diagnostic uncertainty. For example, mutations in PPARG cause Mendelian lipodystrophy and increase risk of type 2 diabetes (T2D). Although approximately 1 in 500 people harbor missense variants in PPARG, most are of unknown consequence. To prospectively characterize PPAR&#x3b3; variants, we used highly parallel oligonucleotide synthesis to construct a library encoding all 9,595 possible single-amino acid substitutions. We developed a pooled functional assay in human macrophages, experimentally evaluated all protein variants, and used the experimental data to train a variant classifier by supervised machine learning. When applied to 55 new missense variants identified in population-based and clinical sequencing, the classifier annotated 6 variants as pathogenic; these were subsequently validated by single-variant assays. Saturation mutagenesis and prospective experimental characterization can support immediate diagnostic interpretation of newly discovered missense variants in disease-related genes."

    response = get_inference_summary_prompt(gene=gene, text_input=data, max_tokens=512, log=True)
    print("got inference: \n{}".format(response))


    # get the connection
    conn = dcc_gpt_lib.get_connection()

    # get the name and prompt of the run
    _, name_run, prompt_run = dcc_gpt_lib.get_db_run_data(conn=conn, id_run=id_run)
    print("got run: {} with prompt: \n'{}'\n".format(name_run, prompt_run))

    # get the list of searches
    list_searches = dcc_gpt_lib.get_db_list_ready_searches(conn=conn, num_searches=1)

    # get the best abstracts for gene
    for search in list_searches:
        id_search = search.get('id')
        id_top_level_abstract = -1
        gene = search.get('gene')

        # test data
        gene = "PPARG"
        id_search = 1

        # log
        print("\nprocessing search: {} for gene: {} for run id: {} of name: {}".format(id_search, gene, id_run, name_run))
        # time.sleep(5)

        # get all the abstracts for the document level and run
        list_abstracts = dcc_gpt_lib.get_list_abstracts(conn=conn, id_search=id_search, id_run=id_run, num_level=num_level, num_abstracts=max_per_level, log=True)

        for i in range(0, len(list_abstracts), num_abstracts_per_summary):
            list_sub = list_abstracts[i : i + num_abstracts_per_summary] 

            # for the sub list
            str_abstracts = ""
            for item in list_sub:
                abstract = item.get('abstract')
                word_count = len(abstract.split())
                print("using abstract: {} with word count: {} and ref count: {}".format(item.get('id'), word_count, item.get('ref_count')))
                str_abstracts = str_abstracts + "\n" + abstract

            # get the inference
            response = get_inference_summary_prompt(gene=gene, text_input=str_abstracts, max_tokens=512, temperature=0.5, log=True)
            print("got inference: \n{}".format(response))

            # break after one inference
            break





