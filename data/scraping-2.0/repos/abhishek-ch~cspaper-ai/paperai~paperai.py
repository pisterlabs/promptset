from paperai.chain.PaperReader import *
from paperai.chain.UploadInVectorDB import *
from paperai.prompt import PROMPT_TEMPLATE, OBSERVATION_TOKEN
from paperai.agent.CustomAgent import *
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from paperai.llm import ChatLLM



def get_model(model_name:str) -> BaseOpenAI:
    if openai.api_type == "azure":
        return ChatOpenAI(engine=model_name, model_name=gpt_models_dict.get(model_name), temperature=0)
    return OpenAI(temperature=0.7)


def execute(question: str, model:str="gpt-35-turbo") -> dict[str,str]:
    print("****************Execute DB interface*******************")
    # print(f"Question {question}")
    # llm = get_model(model_name=model)
    chatllm = ChatLLM(model=model)
    chain = load_qa_chain(chatllm.llm)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size=1)


    db = DatabaseInterface(embeddings)
    cs_database_chain = CSPaperChain(chain=chain,vectordb=db)
    # Initialize your custom Tool
    cs_paper_tool = Tool(
        name="Computer Science Topics",
        func=cs_database_chain.run,
        description="""
        useful for when you need to answer questions about a Computer Science paper specifically 
        databases, design and architecture.
        The tool is useful for gathering information on various technical subjects related to computer science, 
        including databases, algorithms, data structures, system design, architecture, 
        and other software engineering topics.
        """
    )


    vectordb_chain = UploadInVectorDB(chain=chain, vectordb=db)
    # Initialize your custom Tool
    # TODO Activate this tool after further testing
    # upload_in_vectordb_tool = Tool(
    #     name="Upload Vector Database",
    #     func=vectordb_chain.run,
    #     description="""
    #     useful for when you need to upload documents from a location to the qdrant vector database.
    #     The tool is useful for extracting embeddings and upload it to the
    #     collection cs_papers of  Qdrant vector database
    #     """
    # )

    tools = [cs_paper_tool]

    prompt = PROMPT_TEMPLATE.format(
        tool_description="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools]),
        question=question,
        previous_responses='{previous_responses}',
        )

    # Initialize your Agent
    agent = CustomAgent(
        chatllm=chatllm, 
        tools=tools, 
        prompt=prompt, 
        stop_pattern=[f'\n{OBSERVATION_TOKEN}', f'\n\t{OBSERVATION_TOKEN}'],
        pdf_details={}
    )
    
    try:
        result = agent.run(question)
        # print(result)
        # print(f"PDF References are: {agent.pdf_details.get('source')}")
        # print(f"Another: {agent.pdf_details.get('page')}")
        return {"response": result, "metadata": agent.pdf_details}
    except Exception as err:
        print(f"Crashed++ \n {err}")
    finally:
        print(f"Resetting db....")
        db.reset()