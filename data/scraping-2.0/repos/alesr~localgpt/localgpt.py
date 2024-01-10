from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
import gradio as gr


class ChatbotIndex:
    def __init__(self, model_name, directory_path):
        self.llm_predictor = LLMPredictor(ChatOpenAI(model_name=model_name))
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=self.llm_predictor)
        self.docs = SimpleDirectoryReader(directory_path).load_data()

    def construct_index(self):
        self.index = GPTVectorStoreIndex.from_documents(
            self.docs, service_context=self.service_context)
        self.index.storage_context.persist(persist_dir='index')
        return self.index

    def load_index(self):
        storage_context = StorageContext.from_defaults(persist_dir="index")
        self.index = load_index_from_storage(storage_context)

    def query_response(self, input_text):
        query_engine = self.index.as_query_engine()
        response = query_engine.query(input_text)
        print(response)
        return response.response


def launch_chatbot_interface():
    chatbot = ChatbotIndex(model_name='gpt-3.5-turbo', directory_path="data")
    chatbot.construct_index()

    iface = gr.Interface(fn=chatbot.query_response, inputs="text",
                         outputs="text", title="LocalGPT Chatbot")
    iface.launch(share=True)


if __name__ == "__main__":
    launch_chatbot_interface()
