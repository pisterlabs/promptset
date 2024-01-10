def gen_relfect_prompt_accu(task, log, Legis, Ticker):
    prompt = f"""
    Previously, I presented you with a link prediction task:

    [Previous TASK]-----------------
    {task}
    -------------------------------

    Your response was:

    [LOG]---------------------------
    {log}
    --------------------------------
    Your response was ACCURATE, as there is indeed a "BUY-SELL" relationship between "{Legis}" and "{Ticker}".

    Moving on, your next objective involves self-reflection:

    [TASK]
    Building upon your success with the previous task, take a moment to consider the key factors that contributed to your achievement. Reflect specifically on how and when certain strategies can be applied in future scenarios.

    Then, structure your self-assessment in two parts:

    Condition: Identify specific conditions under which the strategy you previously employed can be effectively utilized.
    Action: Outline the steps of the strategy to be employed under these conditions.
    This analysis will serve as a guide for tackling similar tasks in the future, where "similarity" is determined by the conditions you delineate. You will be provided with a new task later on, along with a tool to detect conditions akin to those you've described by assessing vector similarity, enabling you to retrieve and apply the successful action from your previous experience.

    For example:

        [Example 1]
            Condition: Scenarios where a legislator has multiple overlapping NAICS codes with the tickers they've engaged in transactions with.
            Action: In such scenarios, confidently assert the existence of a relationship.

        [Example 2]
            Condition: Situations where there are no overlapping NAICS codes between a ticker and the legislator's transactions.
            Action: In these instances, scrutinize any bill titles lobbied by the ticker that are pertinent to the committees the legislator belongs to.

    [Output Format]
    We want your answer to be jsonl format with the list of your pair of "Condition" and "Action" as follows:

    [{{
    "condition": "Instances where a legislator has several overlapping NAICS codes among the tickers they've transacted with.",
    "action": "In such a data landscape, confidently indicate 'EXIST'."
    }}, ... ] # store multiple pairs of "Condition" and "Action" in a list if you have multiple pairs of "Condition" and "Action" to store in your memory.
    """
    return prompt

def gen_reflect_prompt_inaccu(task, log, Legis, Ticker):
    prompt = f"""
    Previously, I presented you with a link prediction task:

    [Previous TASK]-----------------
    {task}
    -------------------------------

    Your response was:

    [LOG]---------------------------
    {log}
    --------------------------------

    Your response was INACCURATE, as there exists a "BUY-SELL" relationship between "{Legis}" and "{Ticker}".

    Moving on, your next objective involves self-reflection:

    [TASK]
    Building upon your failure with the previous task, take a moment to consider the key factors that contributed to your failure. Then reflect on how and when certain strategies can be applied in future scenarios to avoid such failure. 
    Now that you know the answer of the task, you can try multiple possible queries using the tool GraphDB Query to see which features of the data could have helped you to solve the task.

    Then, structure your self-assessment in two parts:

    Condition: Identify specific conditions under which the strategy you came up with reflection employed.
    Action: Outline the steps of the strategy to be employed under these conditions.

    This analysis will serve as a guide for tackling similar tasks in the future, where "similarity" is determined by the conditions you delineate. You will be provided with a new task later on, along with a tool to detect conditions akin to those you've described by assessing vector similarity, enabling you to retrieve and apply the successful action from your previous experience.

    For example:

        [Example 1]
            Condition: Scenarios where a legislator has multiple overlapping NAICS codes with the tickers they've engaged in transactions with.
            Action: In such scenarios, confidently assert the existence of a relationship.

        [Example 2]
            Condition: Situations where there are no overlapping NAICS codes between a ticker and the legislator's transactions.
            Action: In these instances, scrutinize any bill titles lobbied by the ticker that are pertinent to the committees the legislator belongs to.

    However, as you reflect on the task you've just completed, distill a novel condition-action pair that captures a new understanding or strategy not previously recorded in your memory. This pair should be informed by unique aspects or outcomes of the current task. If no new insights have emerged or if the task's conditions do not warrant a fresh approach distinct from existing entries, please return an empty list.


    [Output Format]
    We want your answer to be jsonl format with the list of your pair of "Condition" and "Action" as follows:

    [{{
    "condition": "Instances where a legislator has several overlapping NAICS codes among the tickers they've transacted with.",
    "action": "In such a data landscape, confidently indicate 'EXIST'."
    }}, ... ] # store multiple pairs of "Condition" and "Action" in a list if you have multiple pairs of "Condition" and "Action" to store in your memory.
    """
    return prompt


def reflect(agent, eval, log, task, legislator, ticker):
    if eval == 'Accurate':
        prompt = gen_relfect_prompt_accu(task, log, legislator, ticker)
    elif eval == 'Inaccurate':
        prompt = gen_reflect_prompt_inaccu(task, log, legislator, ticker)
    else:
        raise ValueError(f"eval should be either 'Accurate' or 'Inaccurate' but got {eval}")

    answer = agent(prompt)

    import json
    jsons = json.loads(answer['output'])
    print("reflection result", jsons)

    for item in jsons:
        # insert_reflection(item['condition'], item['action'])
        insert_reflection_w_check(item['condition'], item['action'])

    return True


def insert_reflection(condition, action):
    from uuid import uuid4
    import pinecone

    ids = [str(uuid4())]

    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY = "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"
    # Initialize Pinecone
    # PINECONE_API_KEY = "978bf2d9-c8d3-449f-bdfb-5adeb77a6e97"
    PINECONE_API_KEY = "85623b4c-6d07-4e82-a03c-9b06beb27d88"
    PINECONE_ENV = "gcp-starter"
    from langchain.embeddings.openai import OpenAIEmbeddings

    # Initialize Pinecone
    index_name = "decision-tree"
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.Index(index_name)

    model_name = "text-embedding-ada-002"
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    embeds = embed.embed_documents([condition])

    metadatas = [
        {"condition": condition, "action": action}
    ]  # parent is usally the textual description of the data, child is usually the another text description of data or the decision itself.

    index.upsert(vectors=zip(ids, embeds, metadatas))
    return True
    
def insert_reflection_w_check(condition, action, similarity_threshold=0.95):
    from langchain.embeddings.openai import OpenAIEmbeddings
    from uuid import uuid4
    import pinecone

    ids = [str(uuid4())]

    OPENAI_API_KEY = "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"
    PINECONE_API_KEY = "85623b4c-6d07-4e82-a03c-9b06beb27d88"
    PINECONE_ENV = "gcp-starter"

    # Initialize Pinecone
    index_name = "decision-tree"
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.Index(index_name)

    model_name = "text-embedding-ada-002"
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    embeds = embed.embed_documents([condition])

    # Query Pinecone to check for similar vectors
    query_results = index.query(vector=embeds, top_k=1)

    print("Query results:", query_results)

    # Extract the similarity score of the closest vector
    closest_vector_similarity = query_results["matches"][0]["score"] if query_results["matches"] else 0

    # If the closest vector is less similar than the threshold, insert the new vector
    if closest_vector_similarity < similarity_threshold:
        metadatas = [{"condition": condition, "action": action}]
        index.upsert(vectors=list(zip(ids, [embeds], metadatas)))
        print("Vector inserted.")
        return True, "Vector inserted."
    else:
        print("Vector not inserted due to high similarity.")
        return False, "Vector not inserted due to high similarity."





if __name__ == "__main__":
    # data = [{
    # "condition": "When the industry of the company in question is unknown, but the legislator has had financial transactions with other companies and the company has lobbied bills assigned to the committees the legislator belongs to.",
    # "action": "Check the legislator's financial transactions with other companies to understand their investment behavior, and check if the company has lobbied any bills assigned to the committees the legislator belongs to. This could indicate a potential connection between them and suggest the likelihood of a transaction."
    # }]
    # insert_reflection_w_check(data[0]['condition'], data[0]['action'])
    pass