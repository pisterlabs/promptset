from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import csv
import json

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0.7, request_timeout=120)

qe_prompt = PromptTemplate(
    input_variables=["query"],
    template="""as a culinary expert expand recipe search query '''{query}''' 
    into one descriptive sentence. use culinary creative liberty in interpreting the query.
    
    Provide output in json format with following fields:
    1. expanded_query: recipe search query expanded into descriptive one sentence. return
    empty string if unable to expand the query.
    """
)

qe_chain = LLMChain(llm=llm, prompt=qe_prompt)


def expand_query(query):
    output = qe_chain.run(query)
    try:
        json_obj = json.loads(output)
        json_obj["parse_success"] = True

        return json_obj
    except ValueError:
        return {"parse_success": False,
                "raw_string": output}


def expand_queries(query_file, expanded_query_file):
    with open(query_file) as input_file:
        with open(expanded_query_file, "w") as output_file:
            reader = csv.reader(input_file)

            header = next(reader, None)

            for row in reader:
                query = row[0].strip().lower()
                result = expand_query(query)
                result["query"] = query
                output_file.write(json.dumps(result))
                output_file.write("\n")


def main():
    expand_queries("/Users/agaramit/Downloads/Search Results - solr_sample.csv",
                   "/Users/agaramit/Downloads/expanded_queries.json")


if __name__ == '__main__':
    main()
