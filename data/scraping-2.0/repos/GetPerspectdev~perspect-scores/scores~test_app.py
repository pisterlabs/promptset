from app import One_Class_To_Rule_Them_All
from langchain import LlamaCpp, OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import tiktoken
from sentence_transformers import SentenceTransformer
import time
import sys


if __name__ == "__main__":
    #with open(f'./test_out.txt', 'w+', encoding="utf-8") as sys.stdout:
    t0 = time.time()

    the_finger = CallbackManager([StreamingStdOutCallbackHandler])
    the_size = 8000

    the_ring = LlamaCpp(
        model_path="./scores/test_model/mistral-7b-instruct-v0.1.Q4_0.gguf",
        temperature=0.75,
        max_tokens=100,
        top_p=1,
        callback_manager=the_finger,
        verbose=True,
        n_ctx=32000,
        n_gpu_layers=100,
        n_batch=512,
        n_threads=1,
        seed=8855,
    )


    the_forge = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    scorer = One_Class_To_Rule_Them_All(
        and_in=the_ring,
        the="./scores/data",
        darkness="./scores/indexes",
        bind=True,
        them=the_forge
    )

    # scorer.load_ds_and_idx(1)
    # print("Loaded embeddings")

    # obj = scorer.professionalism_score(slack_token=bennett_slack_token, user_id="U03FGKV2ML")
    # #print(obj)
    # print("professionalism secure")

    
    scorer.load_ds_and_idx(0)
    scorer.designpatterns_local_score("https://@github.com/IMJONEZZ/NLP.git")

    # repo_name = "perspect-scores"

    # repos = scorer._get_git(bennett_github_token)
    # for repo in repos:
    #     design_obj = scorer.designpatterns_local_score(repo_url=repo[0])
    #     print(f"Design patterns secure for {repo[2]}")
        # archetype_obj = scorer.archetype_score(user_token=bennett_github_token, repo_name=repo[2])
        # print("archetype secure")
        # print(f"Design: {design_obj}\nArch: {archetype_obj}")
    #obj = scorer.designpatterns_local_score(repo_url="https://@github.com/narfdre/Tug-a-Sphero.git")


    

    # repos = scorer._get_git(github_token)
    # for repo in repos:
    #     try:
    #         print(repo[2])
    #         obj = scorer.archetype_score(user_token=github_token, repo_name=repo[2])
    #         print(obj)
    #         print("archetype secure")
    #     except Exception as e:
    #         print(e)

    t1 = time.time()
    total = t1-t0
    print(f"Total Time to run: {total}")