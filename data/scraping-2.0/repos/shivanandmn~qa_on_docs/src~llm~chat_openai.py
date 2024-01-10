from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv
import openai


_ = load_dotenv(find_dotenv())


def load_llm(vector_db, config):
    llm = ChatOpenAI(model=config.get("llm_model_name", "gpt-3.5-turbo"), temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=vector_db.as_retriever(search_type="mmr")
    )
    return qa_chain


def load_llm_completion(prompt):
    out = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )["choices"][0]["message"]["content"]
    # out = llm()
    return out


if __name__ == "__main__":
    out = load_llm_completion({})
    print(out)
