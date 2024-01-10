from pathlib import Path
import argparse
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

from document_store.faiss_store import FaissStore
from chain.Educhat import Educhat
from chain.prompt_template.prompt_template import answer_prompt

if __name__ == '__main__':
    llm = Educhat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--document_path", type=str, required=True, help="path to the document")
    args = parser.parse_args()
    faiss_store = FaissStore(Path(args.document_path))


    memory = ConversationBufferMemory(memory_key="chat_history",
                                      input_key="human_input",
                                      return_messages=True)
    chain = load_qa_chain(
        llm, chain_type="stuff", memory=memory, prompt=answer_prompt, verbose=True
    )
    while True:
        query = input("请输入你的问题：")
        docs = faiss_store.get_relevant_documents(query)
        result = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
        print(result)
        # print(chain.memory.buffer)
