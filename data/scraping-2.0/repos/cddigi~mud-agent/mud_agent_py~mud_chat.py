import logging
import os

from anthropic import AI_PROMPT, HUMAN_PROMPT

import claude_retriever
import config

config.setup()

logger = logging.getLogger(__name__)
ANTHROPIC_SEARCH_MODEL = os.environ["ANTHROPIC_SEARCH_MODEL"]

MUD_SEARCH_TOOL_DESCRIPTION = "The search engine will seach over the MudBlazor documentation database and return relevant information and examples using ```csharp``` tags."
from claude_retriever.searcher.searchtools.embeddings import EmbeddingSearchTool


def _is_text_file(filename):
    with open(filename, "rb") as f:
        import chardet

        contents = f.read()
        encoding = chardet.detect(contents)["encoding"]
        if encoding is None:
            return False
        return encoding.startswith("UTF")  # or encoding.startswith('ISO')


def _find_files(input_dir):
    import os

    file_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if _is_text_file(os.path.join(root, file)):
                with open(os.path.join(root, file), "r") as f:
                    try:
                        file_contents = f.read().replace("\n", "")
                        file_list.append({"metadata": file, "text": file_contents})
                    except:
                        pass
    return file_list


def _write_jsonl(file_list, output_file):
    with open(output_file, "w") as f:
        import json

        for file in file_list:
            json.dump(file, f)
            f.write("\n")

    logger.info("Wrote JSONL file to %s", output_file)
    return output_file


def _create_vector_store(vector_store):
    from claude_retriever.utils import embed_and_upload

    if len(vector_store.embeddings) == 0:
        logger.info("Vector store is empty. Filling it from local text files.")
        file_name = "mud-blazor"
        file_list = _find_files(f"data/{file_name}")
        jsonl_file = f"data/{file_name}.jsonl"
        jsonl_file = _write_jsonl(file_list, jsonl_file)
        batch_size = 128

        embed_and_upload(
            input_file=jsonl_file, vectorstore=vector_store, batch_size=batch_size
        )


def _system_prompt():
    system_prompt = f"""
    You are a friendly dotnet programming assistant with a focus on Blazor using the MudBlazor framework. 
    Please write a response to the user that answers their query and provides them with helpful feedback. 
    Feel free to use the search results above to help you write your response, or ignore them if they are not helpful.

    Please ensure your results are in the following format:

    Your response to the user's query.

    ```csharp
    @code{{
      Relevent Blazor code
    }}
    ```
    """

    return system_prompt


def search_prompt(query, search_results):
    prompt = f"""{HUMAN_PROMPT}
    Here are a set of search results that might be helpful for answering the user's query provided by the MudBlazor Agent:

    <result>
    {search_results}
    </result>

    Once again, here is the user's query:

    <query>{query}</query>
    {AI_PROMPT}
    """
    return prompt


def agent():
    from claude_retriever.searcher.vectorstores.local import LocalVectorStore

    store_path = "data/mud-blazor-embeddings.jsonl"
    vector_store = LocalVectorStore(store_path)
    _create_vector_store(vector_store)
    search_tool = EmbeddingSearchTool(
        tool_description=MUD_SEARCH_TOOL_DESCRIPTION, vector_store=vector_store
    )
    client = claude_retriever.ClientWithRetrieval(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        search_tool=search_tool,
    )
    return client


def search(query):
    search_results = agent().retrieve(
        query=query,
        stop_sequences=[HUMAN_PROMPT, "END_OF_SEARCH"],
        model=ANTHROPIC_SEARCH_MODEL,
        n_search_results_to_use=3,
        max_searches_to_try=5,
        max_tokens_to_sample=2000,
    )

    return search_results


def main():
    completion_list = []
    history_str = _system_prompt()
    query = ""

    while query != "exit":
        query = input("Enter question (or exit): ")
        completion_list.append({"query": query, "response": None})

        if query == "exit":
            break

        search_results = search(query)
        history_str = "\n".join(
            [
                f"Query: {c['query']} \n Response: {c['response']}"
                for c in completion_list
            ]
        )

        prompt = search_prompt(query, search_results)
        prompt = f"{history_str}\n\n{prompt}"
        response = agent().completions.create(
            prompt=prompt,
            stop_sequences=[HUMAN_PROMPT],
            model=ANTHROPIC_SEARCH_MODEL,
            max_tokens_to_sample=4000,
            temperature=0.8,
        )

        logger.info("-" * 50)
        logger.info("Response:")
        logger.info(response.completion)
        logger.info("-" * 50)
        logger.debug(response.completion)


if __name__ == "__main__":
    main()
