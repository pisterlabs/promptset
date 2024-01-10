import os

from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableBranch
from langchain.prompts import PromptTemplate
from langchain.chains import create_extraction_chain

from llm_helpers.openai_prompts import descisionprompt, recommedation_titels_prompt, general_prompt, recro_prompt, question_answering_prompt, extract_title_prompt
from tmdb.helper import get_recro_movies, get_detail_moviedata


def get_chain():

    config="test"

    # Setup keys
    openai_api_key = os.environ['openai_api_key']

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        max_tokens=1000
    )

    llm_gpt4_temp = ChatOpenAI(
        model="gpt-4",
        openai_api_key=openai_api_key,
        max_tokens=1000,
        temperature=0.3,
    )
    

    llm_gpt4 = ChatOpenAI(
        model="gpt-4",
        openai_api_key=openai_api_key,
        max_tokens=1000,        
    )
    
    
    openai_functioncall_to_get_the_titles = create_functioncall_chain_for_titles(openai_api_key)



    #define chains
    descissionchain = ( descisionprompt | llm | StrOutputParser() )
    #recommendation_chain = (recommedation_titels_prompt | llm )

#{"titles":StrOutputParser(), "config":config }|

    # recommendation_chain_alt = (
    #     {"context": recommedation_titels_prompt | llm | StrOutputParser() | RunnableLambda(_get_movie_data), "input":RunnablePassthrough(), "lang":RunnablePassthrough()}
    #     | recro_prompt
    #     | llm
    #     | StrOutputParser()
    # )


    recommendation_chain = (
       
        {"context": {"titles": recommedation_titels_prompt | llm_gpt4_temp | StrOutputParser() | {"input": RunnablePassthrough()} | openai_functioncall_to_get_the_titles , "config":RunnablePassthrough() }| RunnableLambda(get_recro_movies), "input":RunnablePassthrough(), "lang":RunnablePassthrough()}
        | recro_prompt
        | llm_gpt4
        | StrOutputParser()
    )


    info_chain = (
        {"context": extract_title_prompt | llm |StrOutputParser() | RunnableLambda(get_detail_moviedata), "input":RunnablePassthrough()}
        | question_answering_prompt
        | llm
        | StrOutputParser())


    general_chain = (general_prompt | llm)


    # stick everything together    
    branch = RunnableBranch(
        (lambda x: "recommendation" in x["topic"].lower(), recommendation_chain),
        (lambda x: "question" in x["topic"].lower(), info_chain),
        general_chain,
    )
    full_chain = {"topic": descissionchain, "input": lambda x: x["input"], "lang":lambda x: x["lang"], "provider":lambda x: x["provider"]} | branch

    return full_chain




def create_functioncall_chain_for_titles(openai_api_key):

  template = """
    You have to extract the movie titles form given input and return them as a Python List of strings

    input: {input}
    output:
    """

  prompt_template = PromptTemplate(input_variables=["input"], template=template)

  # Schema
  schema = {
      "properties": {
          "movie_title_{i}": {"type": "string"},
      },
      "required": ["movie_title"],
  }

  #create chain
  llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",  openai_api_key=openai_api_key)
  chain = create_extraction_chain(schema=schema, llm=llm, verbose=True, prompt=prompt_template)
  #chain = create_extraction_chain(schema=schema, llm=llm, verbose=True )
  return chain
