from mongoQuery import get_result
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def get_llm_result(query):
    print("Start get llm result")
    
    res = get_result(query)
    user_query = query
    
    conversation_1 = res[0]["Text"]
    conversation_2 = res[1]["Text"]
    conversation_3 = res[2]["Text"]

    summary_prompt = [
        {"role": "system", "content": "You are a helpful legal from Harvard helping the jury recall testimony from a trial or deposition. The jury has a question about the trial/deposition and you are tasked with finding the relevant information."},
        {"role": "user", "content": f"The jury has the following question: {user_query}. You are given 3 relevant conversations from vector embeddings, with each conversation having 12 sentences. The first conversation is {conversation_1}, the second conversation is {conversation_2}, and the third conversation is {conversation_3}. You should read each conversation and determine which sentences in each conversation is most relevan to the jury's question. You should then return the relevant sentences for each conversation, ensuring continuity (e.g. sentence 1-10, 2-9, 3-11, etc.) You cannot alter any of the sentences in the conversation as it is spoken words. On top of that, formulate a strong argument using those conversations that help the user prove their point. The response should be clean and cohesive."}
    ]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=summary_prompt
    )
    response_res = response.choices[0].message.content
    print("LLM response OK")
    # print(f'LLM response:{response_res}')
    return response_res
