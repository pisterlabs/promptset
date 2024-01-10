import pickle
import numpy as np
import openai
import tiktoken
import pandas as pd


COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
	result = openai.Embedding.create(
	  model=model,
	  input=text
	)
	return result["data"][0]["embedding"]



def vector_similarity(x: list[float], y: list[float]) -> float:
	"""
	Returns the similarity between two vectors.
	
	Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
	"""
	return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
	"""
	Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
	to find the most relevant sections. 
	
	Return the list of document sections, sorted by relevance in descending order.
	"""
	query_embedding = get_embedding(query)
	
	document_similarities = sorted([
		(vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
	], reverse=True)
	
	return document_similarities


MAX_SECTION_LEN = 2500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
	"""
	Fetch relevant 
	"""
	most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

	chosen_sections = []
	chosen_sections_len = 0
	chosen_sections_indexes = []
	 
	for _, section_index in most_relevant_document_sections:
		# Add contexts until we run out of space.        
		document_section = df.loc[section_index]
		# print(document_section)
		
		chosen_sections_len += document_section.tokens + separator_len
		if chosen_sections_len > MAX_SECTION_LEN and len(chosen_sections) > 0:
			# print('breaking')
			break
			
		chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
		print(chosen_sections)
		chosen_sections_indexes.append(str(section_index))
			
	# Useful diagnostic information
	# print(f"Selected {len(chosen_sections)} document sections:")
	# print("\n".join(chosen_sections_indexes))
	
	header = """Answer the question as truthfully as possible using the provided context, don't answer in the first person, use full names, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
	
	return { 'prompt': header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:", 'docs': most_relevant_document_sections }


COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    docs = prompt['docs']
    prompt = prompt['prompt']
    
    if show_prompt:
        print(prompt)

    
    response = openai.Completion.create(
                prompt=prompt,
                timeout=5.0,
                **COMPLETIONS_API_PARAMS
            )

    return { 'response' : response["choices"][0]["text"].strip(" \n"), 'docs' :docs}






def lambda_handler(event, context):



	df = pd.read_csv('docs.csv')
	df = df.set_index(["title", "heading"])

	file = open("embeddings.binary",'rb')
	document_embeddings = pickle.load(file)
	file.close()
	print(event)
	results = answer_query_with_context(event['queryStringParameters']['q'], df, document_embeddings)
	docs = []

	for doc in results['docs']:
		docs.append(doc[1][1])
		if len(docs) >=10:
			break
	results['docs'] = docs
	return results



