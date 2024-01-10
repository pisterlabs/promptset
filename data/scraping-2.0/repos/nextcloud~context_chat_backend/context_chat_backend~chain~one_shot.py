from langchain.llms.base import LLM

from ..vectordb import BaseVectorDB

_LLM_TEMPLATE = '''Answer based only on this context and do not add any imaginative details:
{context}

{question}
'''


def process_query(
	user_id: str,
	vectordb: BaseVectorDB,
	llm: LLM,
	query: str,
	use_context: bool = True,
	ctx_limit: int = 5,
	template: str = _LLM_TEMPLATE,
) -> (str, list):
	if not use_context:
		return llm.predict(query), []

	user_client = vectordb.get_user_client(user_id)
	if user_client is None:
		return llm.predict(query), []

	context_docs = user_client.similarity_search(query, k=ctx_limit)
	context_text = '\n\n'.join(map(
		lambda d: f'{d.metadata.get("title")}\n{d.page_content}',
		context_docs,
	))

	output = llm.predict(template.format(context=context_text, question=query)).strip()
	unique_sources = list(set(map(lambda d: d.metadata.get('source', ''), context_docs)))

	return output, unique_sources
