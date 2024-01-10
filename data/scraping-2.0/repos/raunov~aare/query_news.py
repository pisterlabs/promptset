from dotenv import find_dotenv, load_dotenv
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv(find_dotenv())

company = "LHV_Group"
llm_model = "gpt-3.5-turbo"

def query_news(query_str: str, company: str, llm_model="gpt-3.5-turbo") -> str:
    from llama_index import StorageContext, load_index_from_storage, LLMPredictor, ServiceContext, QuestionAnswerPrompt
    from langchain.chat_models import PromptLayerChatOpenAI

    # new version of llama index uses StorageContext instead of load_from_disk
    # index = GPTSimpleVectorIndex.load_from_disk('index_news.json')
    storage_context = StorageContext.from_defaults(persist_dir=f"storage/{company}")
    # LLM Predictor (gpt-3.5-turbo) + service context
    llm_predictor = LLMPredictor(llm=PromptLayerChatOpenAI(temperature=0, model_name=llm_model))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    index = load_index_from_storage(storage_context,service_context=service_context)

    qa_prompt_template = (
        "Olemasolev info on järgmine: \n"
        "------------------------------------\n"
        "{context_str}"
        "\n----------------------------------\n"
        "Olemasolevat infot arvestades ja mitte varasemat teadmist kasutades vasta küsimusele:"
        "\n{query_str}\n"
    )
    qa_prompt = QuestionAnswerPrompt(qa_prompt_template)

    # new version of llama index uses query_engine.query()
    query_engine = index.as_query_engine(text_qa_template=qa_prompt)
    response = query_engine.query(query_str)
    service_context.llama_logger.get_logs()
    
    return response