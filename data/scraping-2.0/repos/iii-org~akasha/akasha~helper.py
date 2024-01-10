import numpy as np
import jieba
import json
from pathlib import Path
import opencc
from typing import Callable, Union
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from akasha.models.hf import chatGLM, get_hf_model, custom_model, custom_embed
from akasha.models.llama2 import peft_Llama2, get_llama_cpp_model
import os
import shutil

jieba.setLogLevel(jieba.logging.INFO)  ## ignore logging jieba model information


def del_path(path, tag="temp_c&r@md&"):
    p = Path(path)
    for file in p.glob("*"):
        if tag in file.name:
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)

    return


def is_path_exist(path: str) -> bool:
    try:
        des_path = Path(path)
        if not des_path.exists():
            raise FileNotFoundError("can not find the path")
    except FileNotFoundError as err:
        print(err, path)
        return False
    return True


def _separate_name(name: str):
    """separate type:name by ':'

    Args:
        **name (str)**: string with format "type:name" \n

    Returns:
        (str, str): res_type , res_name
    """
    sep = name.split(":")
    if len(sep) != 2:
        ### if the format type not equal to type:name ###
        res_type = sep[0].lower()
        res_name = ""
    else:
        res_type = sep[0].lower()
        res_name = sep[1]

    return res_type, res_name


def _handle_azure_env() -> (str, str, str):
    """from environment variable get the api_base, api_key, api_version

    Returns:
        (str, str, str): api_base, api_key, api_version
    """
    check_env, ret, count = ["BASE", "KEY", "VERSION"], ["", "", ""], 0
    try:
        for check in check_env:
            if f"AZURE_API_{check}" in os.environ:
                ret[count] = os.environ[f"AZURE_API_{check}"]
            elif f"OPENAI_API_{check}" in os.environ:
                ret[count] = os.environ[f"OPENAI_API_{check}"]
            else:
                if check == "VERSION":
                    ret[count] = "2023-05-15"
                else:
                    raise Exception(
                        f"can not find the openai {check} in environment variable.\n\n"
                    )
            count += 1
    except Exception as err:
        print(err)

    return ret[0], ret[1], ret[2]


def handle_embeddings(embedding_name: str, verbose: bool) -> vars:
    """create model client used in document QA, default if openai "gpt-3.5-turbo"
        use openai:text-embedding-ada-002 as default.
    Args:
        **embedding_name (str)**: embeddings client you want to use.
            format is (type:name), which is the model type and model name.\n
            for example, "openai:text-embedding-ada-002", "huggingface:all-MiniLM-L6-v2".\n
        **logs (list)**: list that store logs\n
        **verbose (bool)**: print logs or not\n

    Returns:
        vars: embeddings client
    """

    if isinstance(embedding_name, Callable):
        embeddings = custom_embed(func=embedding_name)
        if verbose:
            print("selected custom embedding.")
        return embeddings

    embedding_type, embedding_name = _separate_name(embedding_name)

    if embedding_type in ["text-embedding-ada-002", "openai", "openaiembeddings"]:
        import openai

        if (
            "AZURE_API_TYPE" in os.environ and os.environ["AZURE_API_TYPE"] == "azure"
        ) or (
            "OPENAI_API_TYPE" in os.environ and os.environ["OPENAI_API_TYPE"] == "azure"
        ):
            embedding_name = embedding_name.replace(".", "")
            api_base, api_key, api_version = _handle_azure_env()
            embeddings = OpenAIEmbeddings(
                deployment=embedding_name,
                openai_api_base=api_base,
                api_key=api_key,
                openai_api_type="azure",
                api_version=api_version,
            )
        else:
            openai.api_type = "open_ai"

            embeddings = OpenAIEmbeddings(
                model=embedding_name,
                openai_api_base="https://api.openai.com/v1",
                api_key=os.environ["OPENAI_API_KEY"],
                openai_api_type="open_ai",
            )
        info = "selected openai embeddings.\n"

    elif embedding_type in ["rerank", "re"]:
        if embedding_name == "":
            embedding_name = "BAAI/bge-reranker-base"

        embeddings = "rerank:" + embedding_name
        info = "selected rerank embeddings.\n"
    elif embedding_type in [
        "huggingface",
        "huggingfaceembeddings",
        "transformers",
        "transformer",
        "hf",
    ]:
        from langchain.embeddings import (
            HuggingFaceEmbeddings,
            SentenceTransformerEmbeddings,
        )

        embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
        info = "selected hugging face embeddings.\n"

    elif embedding_type in [
        "tf",
        "tensorflow",
        "tensorflowhub",
        "tensorflowhubembeddings",
        "tensorflowembeddings",
    ]:
        from langchain.embeddings import TensorflowHubEmbeddings

        embeddings = TensorflowHubEmbeddings()
        info = "selected tensorflow embeddings.\n"

    else:
        embeddings = OpenAIEmbeddings()
        info = "can not find the embeddings, use openai as default.\n"

    if verbose:
        print(info)
    return embeddings


def handle_model(
    model_name: Union[str, Callable], verbose: bool, temperature: float = 0.0
) -> vars:
    """create model client used in document QA, default if openai "gpt-3.5-turbo"

    Args:
       ** model_name (str)**: open ai model name like "gpt-3.5-turbo","text-davinci-003", "text-davinci-002"\n
        **logs (list)**: list that store logs\n
        **verbose (bool)**: print logs or not\n

    Returns:
        vars: model client
    """
    if isinstance(model_name, Callable):
        model = custom_model(func=model_name, temperature=temperature)
        if verbose:
            print("selected custom model.")
        return model

    model_type, model_name = _separate_name(model_name)

    if model_type in ["openai", "openaiembeddings"]:
        import openai

        if (
            "AZURE_API_TYPE" in os.environ and os.environ["AZURE_API_TYPE"] == "azure"
        ) or (
            "OPENAI_API_TYPE" in os.environ and os.environ["OPENAI_API_TYPE"] == "azure"
        ):
            model_name = model_name.replace(".", "")
            api_base, api_key, api_version = _handle_azure_env()
            model = AzureChatOpenAI(
                deployment_name=model_name,
                temperature=temperature,
                base_url=api_base,
                api_key=api_key,
                api_version=api_version,
            )
        else:
            openai.api_type = "open_ai"
            model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.environ["OPENAI_API_KEY"],
            )
        info = f"selected openai model {model_name}.\n"

    elif model_type in ["remote", "server"]:
        import openai

        openai.api_type = "open_ai"
        if "REMOTE_API_BASE" in os.environ:
            base_url = os.environ["REMOTE_API_BASE"]
        elif "OPENAI_API_BASE" in os.environ:
            base_url = os.environ["OPENAI_API_BASE"]
        else:
            base_url = ""
            print("can not find the openai {check} in environment variable.\n\n")
        model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key="EMPTY",
            base_url=base_url,
        )
        info = f"selected remote model {model_name}.\n"

    elif (
        model_type in ["llama-cpu", "llama-gpu", "llama", "llama2", "llama-cpp"]
        and model_name != ""
    ):
        model = get_llama_cpp_model(model_type, model_name, temperature)
        info = "selected llama-cpp model\n"
    elif model_type in [
        "huggingface",
        "huggingfacehub",
        "transformers",
        "transformer",
        "huggingface-hub",
        "hf",
    ]:
        model = get_hf_model(model_name, temperature)
        info = f"selected huggingface model {model_name}.\n"

    elif model_type in ["chatglm", "chatglm2", "glm"]:
        model = chatGLM(model_name=model_name, temperature=temperature)
        info = f"selected chatglm model {model_name}.\n"

    elif model_type in ["lora", "peft"]:
        model = peft_Llama2(model_name_or_path=model_name, temperature=temperature)
        info = f"selected peft model {model_name}.\n"
    else:
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        info = f"can not find the model {model_type}:{model_name}, use openai as default.\n"
        print(info)
    if verbose:
        print(info)

    return model


def handle_search_type(search_type: str, verbose: bool = False) -> str:
    if callable(search_type):
        search_type_str = search_type.__name__

    else:
        search_type_str = search_type

    # if verbose:
    #     print("search type is :", search_type_str)

    return search_type_str


def get_doc_length(language: str, text:str) -> int:
    """calculate the length of terms in a giving Document

    Args:
        **language (str)**: 'ch' for chinese and 'en' for others, default 'ch'\n
        **doc (Document)**: Document object\n

    Returns:
        doc_length: int Docuemtn length
    """
    if language == "ch":
        doc_length = len(list(jieba.cut(text)))
    else:
        doc_length = len(text.split())
    return doc_length


def get_docs_length(language: str, docs: list) -> int:
    """calculate the total length of terms in giving documents

    Args:
        language (str): 'ch' for chinese and 'en' for others, default 'ch'\n
        docs (list): list of Documents\n

    Returns:
        docs_length: int total Document length
    """
    docs_length = 0
    for doc in docs:
        docs_length += get_doc_length(language, doc.page_content)
    return docs_length


def get_question_from_file(path: str, question_type: str):
    """load questions from file and save the questions into lists.
    a question list include the question, mutiple options, and the answer (the number of the option),
      and they are all separate by space in the file.

    Args:
        **path (str)**: path of the question file\n

    Returns:
        list: list of question list
    """
    f_path = Path(path)
    with f_path.open(mode="r", encoding="utf-8") as file:
        content = file.read()
    questions = []
    answers = []

    if question_type.lower() == "essay":
        content = content.split("\n\n")
        for i in range(len(content)):
            if content[i] == "":
                continue
            process = "".join(content[i].split("問題：")).split("答案：")

            questions.append(process[0])
            answers.append(process[1])
        return questions, answers

    for con in content.split("\n"):
        if con == "":
            continue
        questions.append([word for word in con.split("\t") if word != ""])
    return questions, answers


def extract_result(response: str):
    """to prevent the output of llm format is not what we want, try to extract the answer (digit) from the llm output

    Args:
        **response (str)**: llm output\n

    Returns:
        int: digit of answer
    """
    try:
        res = str(json.loads(response)["ans"]).replace(" ", "")

    except:
        res = -1
        for c in response:
            if c.isdigit():
                res = c

                break
    return res


def get_all_combine(
    embeddings_list: list,
    chunk_size_list: list,
    model_list: list,
    topK_list: list,
    search_type_list: list,
) -> list:
    """record all combinations of giving lists

    Args:
        **embeddings_list (list)**: list of embeddings(str)\n
        **chunk_size_list (list)**: list of chunk sizes(int)\n
        **model_list (list)**: list of models(str)\n
        **topK_list (list)**: list of topK(int)\n
        **search_type_list (list)**: list of search types(str)\n

    Returns:
        list: list of tuples of all different combinations
    """
    res = []
    for embed in embeddings_list:
        for chk in chunk_size_list:
            for mod in model_list:
                for tK in topK_list:
                    for st in search_type_list:
                        res.append((embed, chk, mod, tK, st))

    return res


def get_best_combination(result_list: list, idx: int) -> list:
    """input list of tuples and find the greatest tuple based on score or cost-effective (index 0 or index 1)
    tuple looks like (score, cost-effective, embeddings, chunk size, model, topK, search type)

    Args:
        **result_list (list)**: list of tuples that save the information of running experiments\n
        **idx (int)**: the index used to find the greatest result 0 is based on score and 1 is based on cost-effective\n

    Returns:
        list: return list of tuples that have same highest criteria
    """
    res = []
    sorted_res = sorted(result_list, key=lambda x: x[idx], reverse=True)
    max_score = sorted_res[0][idx]
    for tup in sorted_res:
        if tup[idx] < max_score:
            break
        res_str = (
            "embeddings: "
            + tup[-5]
            + ", chunk size: "
            + str(tup[-4])
            + ", model: "
            + tup[-3]
            + ", topK: "
            + str(tup[-2])
            + ", search type: "
            + tup[-1]
            + "\n"
        )
        print(res_str)
        res.append(tup[-5:])

    return res


def sim_to_trad(text: str) -> str:
    """convert simplified chinese to traditional chinese

    Args:
        **text (str)**: simplified chinese\n

    Returns:
        str: traditional chinese
    """
    cc = opencc.OpenCC("s2t.json")
    return cc.convert(text)


def _get_text(
    texts: list, previous_summary: str, i: int, max_doc_len: int, language:str="ch"
) -> (int, str, int):
    """used in summary, combine chunks of texts into one chunk that can fit into llm model

    Args:
        texts (list): chunks of texts
        previous_summary (str): _description_
        i (int): start from i-th chunk
        max_doc_len (int): the max doc length we want to fit into llm model at one time
        language (str): 'ch' for chinese and 'en' for others, default 'ch'\n

    Returns:
        (int, str, int): return the total tokens of combined chunks, combined chunks of texts, and the index of next chunk
    """
    cur_count = get_doc_length(language, previous_summary)
    words_len = get_doc_length(language, texts[i])
    cur_text = ""
    while cur_count + words_len < max_doc_len and i < len(texts):
        cur_count += words_len
        cur_text += texts[i] + "\n"
        i += 1
        if i < len(texts):
            words_len = get_doc_length(language,texts[i])

    return cur_count, cur_text, i


def call_model(model, prompt: str) -> str:
    """call llm model and return the response

    Args:
        model (_type_): llm model
        prompt (str): input prompt

    Returns:
        str: llm response
    """
    try:  ### try call openai llm model
        response = model.predict(prompt)

    except:
        response = model._call(prompt)
    return response


def get_non_repeat_rand_int(vis: set, num: int):
    temp = np.random.randint(num)
    if len(vis) >= num:
        vis = set()
    if temp not in vis:
        vis.add(temp)
        return temp
    return get_non_repeat_rand_int(vis, num)


def get_text_md5(text):
    import hashlib

    md5_hash = hashlib.md5(text.encode()).hexdigest()

    return md5_hash


def image_to_base64(image_path: str) -> str:
    """convert image to base64 string

    Args:
        image_path (str): path of image

    Returns:
        str: base64 string
    """
    import base64

    with open(image_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read())
    return img_str.decode("utf-8")
