from concurrent.futures import ThreadPoolExecutor
from . import openai_calls
from . import constants
from . import openai_calls
# from transformers import GPT2Tokenizer

'''
    This function returns list of chunks created considering max_token_length provided by user.
'''
def chunk_text(text, max_token_length):
    sentences = text.split('. ')  # Simple split by sentences
    chunks = []
    current_chunk = ""

    # Adi - We always try to max out token lengths so we can do least number of parallel api calls
    # this way we max out the tokens per gpt4 api call.
    approxNumberOfTotTokens = len(text)/4
    numParallelApiCalls = int(approxNumberOfTotTokens/max_token_length)+1
    perWorkerTokenLength = approxNumberOfTotTokens/numParallelApiCalls
    print("perWorkerTokenLength: ", perWorkerTokenLength)
    print("numParallelApiCalls: ", numParallelApiCalls)
    print("approxNumberOfTotTokens: ", approxNumberOfTotTokens)
    for sentence in sentences:
        # Check if adding the next sentence exceeds the max token length
        # Adi - the GPT2 tokenizer used here is causing token limit errors - replacing num tokens by (num chars)/4 as an approximation
        # legacy - if calculate_token_length(current_chunk + sentence) > perWorkerTokenLength:
        if len(current_chunk + sentence)/4 > perWorkerTokenLength:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence + '. '

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def multithreaded_summarized_json(list_of_chunks,model,query_prompt):
   # Function to call OpenAI API
    def summarize_chunk_to_json(chunk):
        prompt = query_prompt + "\n\n Text: " + chunk + "\n\nJSON:"
        response = openai_calls.ask_chatgpt(prompt, model)
        str_response = str(response)
        # Some sanity text cleaning to avoid errors in yaml loading
        str_response = str_response.replace("json", "")
        str_response = str_response.replace("`", "")
        return str_response

    # Use ThreadPoolExecutor to process chunks in parallel
    with ThreadPoolExecutor(max_workers=constants.NUMBER_OF_WORKERS) as executor:
        list_of_json_summaries = list(executor.map(summarize_chunk_to_json, list_of_chunks))
    

    # Uncomment this for sanity check whether the sequence of json and corresponding json is maintained or not
    # for i in range(len(list_of_json_summaries)):
    #    print("JSON:",list_of_chunks[i],"Summarize JSON chunk:", list_of_json_summaries[i])

    print(len(list_of_json_summaries))
    return list_of_json_summaries