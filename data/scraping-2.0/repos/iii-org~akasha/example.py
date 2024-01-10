import akasha
import akasha.eval as eval
import akasha.summary as summary


query1 = "五軸是甚麼?"
query2 = [
    "西門子自有工廠如何朝工業4.0 發展",
    "詳細解釋「工業4.0 成熟度指數」發展路徑的六個成熟度",
    "根據西門子自有工廠朝工業4.0發展，探討其各項工業4.0的成熟度指標",
]


## used for custom search type, use query_embeds and docs_embeds to determine relevant documents.
# you can add your own variables using log to record, for example log['dd'] = "miao"
def cust(query_embeds, docs_embeds, k: int, relevancy_threshold: float, log: dict):
    from scipy.spatial.distance import euclidean
    import numpy as np

    distance = [
        [euclidean(query_embeds, docs_embeds[idx]), idx]
        for idx in range(len(docs_embeds))
    ]
    distance = sorted(distance, key=lambda x: x[0])

    ## change dist if embeddings not between 0~1
    max_dist = 1
    while max_dist < distance[-1][0]:
        max_dist *= 10
        relevancy_threshold *= 10

    ## add log para
    log["dd"] = "miao"

    return [
        idx for dist, idx in distance[:k] if (max_dist - dist) >= relevancy_threshold
    ]


def test_model(prompt: str):
    import openai
    from langchain.chat_models import ChatOpenAI

    openai.api_type = "open_ai"
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    ret = model.predict(prompt)

    return ret


def test_embed(texts: list) -> list:
    from sentence_transformers import SentenceTransformer

    mdl = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    embeds = mdl.encode(texts, normalize_embeddings=True)

    return embeds


def QA(doc_path="./docs/mic/"):
    qa = akasha.Doc_QA(verbose=False, search_type="svm")

    qa.get_response(
        doc_path=doc_path,
        prompt=query1,
        chunk_size=500,
        record_exp="",
        search_type=cust,
        max_doc_len=1500,
        system_prompt="請你在回答前面加上喵",
    )

    print(qa.response)

    ### you can use qa.logs to get the log of each run, logs is a dictonary, for each run, the key is timestamp and the log of that run is the value.
    # use qa.timestamp_list to get the timestamp of each run, so you can get the log of each run by using it as key.
    # timestamp_list = qa.timestamp_list
    # print(qa.logs[timestamp_list[-1]])
    # print(qa.logs[timestamp_list[-1]]['dd'])   # the variable you add to log in cust function

    response_list = qa.chain_of_thought(doc_path=doc_path, prompt_list=query2)
    print(response_list)
    return qa


qa_obj = QA()


### EVAL ###


### remember the question_type must match the q_file's type
def EVAL(doc_path: str = "./docs/mic/"):
    eva = eval.Model_Eval(
        question_type="single_choice", record_exp="1234test", verbose=True
    )

    b, c = eva.auto_create_questionset(
        doc_path=doc_path,
        question_num=2,
        record_exp="1234test",
        question_type="single_choice",
        chunk_size=850,
    )
    # b,c = eva.auto_create_questionset(doc_path=doc_path, question_num=2, record_exp="1234test", question_type="essay")

    # qs_path = eva.logs[eva.timestamp_list[-1]]['questionset_path']
    print(
        eva.auto_evaluation(
            questionset_file="questionset/mic_3.txt",
            doc_path=doc_path,
            verbose=True,
        )
    )
    # print(eva.__dict__)

    # eva.optimum_combination(q_file="questionset/mic_15.txt", doc_path=doc_path, chunk_size_list=[500]\
    #     ,search_type_list=[cust,"svm","tfidf"])
    return eva


eval_obj = EVAL()
#eval_obj.save_logs("./eva.json")


### SUMMARY ###
def SUM(file_name: str = "./docs/mic/20230531_智慧製造需求下之邊緣運算與新興通訊發展分析.pdf"):
    sum = summary.Summary(
        chunk_size=1000,
        chunk_overlap=40,
        model="openai:gpt-3.5-turbo",
        verbose=False,
        threshold=0.2,
        language="ch",
        record_exp="1234test",
        format_prompt="請你在回答前面加上喵",
        system_prompt="用中文回答",
        max_doc_len=1500,
        temperature=0.0,
    )

    sum.summarize_file(
        file_path=file_name, summary_type="map_reduce", summary_len=100, verbose=True
    )
    print(sum.logs)
    return sum


summary_obj = SUM()

# summary_obj.save_logs("./summary_logs.json")
# summary_obj.save_logs(file_type="txt")


### VISION ###

# ret = akasha.openai_vision(pic_path=["C:/Users/ccchang/Pictures/oruya.png"], prompt="這張圖是甚麼意思?")

# print(ret)
